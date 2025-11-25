import { useQuery } from '@tanstack/react-query';
import { 
  BarChart3, TrendingUp, TrendingDown, Target, 
  Brain, Clock, PieChart, Activity
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart as RechartsPie, Pie, Cell, Legend
} from 'recharts';
import { authFetch } from '@/lib/auth';

interface AnalyticsData {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_profit_loss: number;
  best_pair: string;
  worst_pair: string;
  ml_accuracy: number;
  avg_trade_duration: string;
  strategy_performance: Record<string, {
    trades: number;
    wins: number;
    win_rate: number;
    profit_loss: number;
  }>;
  pair_performance: Record<string, {
    trades: number;
    wins: number;
    win_rate: number;
    profit_loss: number;
  }>;
  daily_performance: Array<{
    date: string;
    trades: number;
    wins: number;
    profit_loss: number;
  }>;
  trade_distribution: {
    buy: number;
    sell: number;
  };
}

interface PairProgress {
  pair: string;
  closed_trades: number;
  required_trades: number;
  is_trained: boolean;
  model_version: number;
  accuracy: number;
  progress_percent: number;
}

interface MLStatus {
  ml_enabled: boolean;
  total_closed_trades: number;
  trades_until_ml: number;
  min_trades_per_pair: number;
  pair_progress: PairProgress[];
  trained_pairs_count: number;
  total_pairs: number;
  model: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    trades_trained_on: number;
    feature_importance: Record<string, number>;
  } | null;
  last_trained: string | null;
  trades_until_retrain: number;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(value);
}

function StatCard({ 
  title, 
  value, 
  icon: Icon, 
  trend,
  description,
  loading 
}: { 
  title: string; 
  value: string; 
  icon: any; 
  trend?: 'up' | 'down' | 'neutral';
  description?: string;
  loading?: boolean;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-8 w-24" />
        ) : (
          <div className="flex items-baseline gap-2">
            <span className={`text-2xl font-bold font-mono ${
              trend === 'up' ? 'text-success' : 
              trend === 'down' ? 'text-destructive' : ''
            }`}>
              {value}
            </span>
            {trend && (
              trend === 'up' ? (
                <TrendingUp className="h-4 w-4 text-success" />
              ) : trend === 'down' ? (
                <TrendingDown className="h-4 w-4 text-destructive" />
              ) : null
            )}
          </div>
        )}
        {description && (
          <p className="text-xs text-muted-foreground mt-1">{description}</p>
        )}
      </CardContent>
    </Card>
  );
}

const COLORS = ['hsl(217, 91%, 60%)', 'hsl(142, 76%, 46%)', 'hsl(38, 92%, 50%)', 'hsl(0, 84%, 60%)', 'hsl(280, 65%, 60%)', 'hsl(340, 82%, 52%)'];

export default function Analytics() {
  const { data: analytics, isLoading: analyticsLoading } = useQuery<AnalyticsData>({
    queryKey: ['/api/analytics/'],
    queryFn: async () => {
      const res = await authFetch('/api/analytics/');
      if (!res.ok) throw new Error('Failed to fetch analytics');
      return res.json();
    },
  });

  const { data: mlStatus, isLoading: mlLoading } = useQuery<MLStatus>({
    queryKey: ['/api/ml/status/'],
    queryFn: async () => {
      const res = await authFetch('/api/ml/status/');
      if (!res.ok) throw new Error('Failed to fetch ML status');
      return res.json();
    },
  });

  const isLoading = analyticsLoading || mlLoading;

  const pairData = analytics?.pair_performance 
    ? Object.entries(analytics.pair_performance).map(([pair, data]) => ({
        name: pair,
        trades: data.trades,
        winRate: data.win_rate,
        pnl: data.profit_loss,
      }))
    : [];

  const strategyData = analytics?.strategy_performance
    ? Object.entries(analytics.strategy_performance).map(([strategy, data]) => ({
        name: strategy,
        trades: data.trades,
        winRate: data.win_rate,
        pnl: data.profit_loss,
      }))
    : [];

  const distributionData = analytics?.trade_distribution
    ? [
        { name: 'Buy', value: analytics.trade_distribution.buy },
        { name: 'Sell', value: analytics.trade_distribution.sell },
      ]
    : [];

  const dailyData = analytics?.daily_performance || [];

  const featureImportance = mlStatus?.model?.feature_importance
    ? Object.entries(mlStatus.model.feature_importance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 8)
        .map(([name, value]) => ({
          name: name.replace(/_/g, ' '),
          value: (value * 100).toFixed(1),
        }))
    : [];

  return (
    <div className="p-6 space-y-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold">Analytics</h1>
        <p className="text-muted-foreground">Performance metrics and AI trading insights</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Trades"
          value={String(analytics?.total_trades ?? 0)}
          icon={BarChart3}
          loading={isLoading}
        />
        <StatCard
          title="Win Rate"
          value={`${(analytics?.win_rate ?? 0).toFixed(1)}%`}
          icon={Target}
          trend={(analytics?.win_rate ?? 0) >= 50 ? 'up' : 'down'}
          description={`${analytics?.winning_trades ?? 0}W / ${analytics?.losing_trades ?? 0}L`}
          loading={isLoading}
        />
        <StatCard
          title="Best Pair"
          value={analytics?.best_pair ?? 'N/A'}
          icon={TrendingUp}
          loading={isLoading}
        />
        <StatCard
          title="ML Accuracy"
          value={`${(analytics?.ml_accuracy ?? 0).toFixed(1)}%`}
          icon={Brain}
          trend={(analytics?.ml_accuracy ?? 0) >= 60 ? 'up' : 'neutral'}
          loading={isLoading}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Daily P/L Performance
            </CardTitle>
            <CardDescription>
              Profit and loss over the last 30 days
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : dailyData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={dailyData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(value) => `$${value}`} />
                  <Tooltip 
                    formatter={(value: number) => [formatCurrency(value), 'P/L']}
                    labelFormatter={(label) => new Date(label).toLocaleDateString()}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="profit_loss" 
                    stroke="hsl(217, 91%, 60%)" 
                    strokeWidth={2}
                    dot={{ fill: 'hsl(217, 91%, 60%)' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-64 text-muted-foreground">
                No data available yet
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PieChart className="h-5 w-5" />
              Trade Distribution
            </CardTitle>
            <CardDescription>
              Buy vs Sell trades breakdown
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : distributionData.some(d => d.value > 0) ? (
              <ResponsiveContainer width="100%" height={250}>
                <RechartsPie>
                  <Pie
                    data={distributionData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {distributionData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Legend />
                  <Tooltip />
                </RechartsPie>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-64 text-muted-foreground">
                No trades yet
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Performance by Pair
            </CardTitle>
            <CardDescription>
              Win rate and P/L for each trading pair
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : pairData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={pairData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis type="number" tickFormatter={(value) => `$${value}`} />
                  <YAxis type="category" dataKey="name" width={80} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(value: number) => formatCurrency(value)} />
                  <Bar 
                    dataKey="pnl" 
                    fill="hsl(217, 91%, 60%)"
                    radius={[0, 4, 4, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-64 text-muted-foreground">
                No pair data available
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Strategy Performance
            </CardTitle>
            <CardDescription>
              Effectiveness of different trading strategies
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : strategyData.length > 0 ? (
              <div className="space-y-4">
                {strategyData.map((strategy) => (
                  <div key={strategy.name} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{strategy.name}</span>
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary">{strategy.trades} trades</Badge>
                        <Badge variant={strategy.pnl >= 0 ? 'default' : 'destructive'}>
                          {formatCurrency(strategy.pnl)}
                        </Badge>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Progress value={strategy.winRate} className="h-2 flex-1" />
                      <span className="text-sm font-mono w-12 text-right">{strategy.winRate.toFixed(0)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-muted-foreground">
                No strategy data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            ML Model Metrics
          </CardTitle>
          <CardDescription>
            Machine learning model performance and feature analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-48 w-full" />
          ) : mlStatus?.ml_enabled && mlStatus.model ? (
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-4">
                <h4 className="font-semibold">Model Performance</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Accuracy</p>
                    <p className="text-2xl font-bold font-mono">{(mlStatus.model.accuracy * 100).toFixed(1)}%</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Precision</p>
                    <p className="text-2xl font-bold font-mono">{(mlStatus.model.precision * 100).toFixed(1)}%</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Recall</p>
                    <p className="text-2xl font-bold font-mono">{(mlStatus.model.recall * 100).toFixed(1)}%</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">F1 Score</p>
                    <p className="text-2xl font-bold font-mono">{(mlStatus.model.f1_score * 100).toFixed(1)}%</p>
                  </div>
                </div>
                <div className="pt-4 border-t space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Trained on</span>
                    <span className="font-mono">{mlStatus.model.trades_trained_on} trades</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Next retrain in</span>
                    <span className="font-mono">{mlStatus.trades_until_retrain} trades</span>
                  </div>
                  {mlStatus.last_trained && (
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Last trained</span>
                      <span className="font-mono">
                        {new Date(mlStatus.last_trained).toLocaleDateString()}
                      </span>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="space-y-4">
                <h4 className="font-semibold">Feature Importance</h4>
                <div className="space-y-3">
                  {featureImportance.map((feature) => (
                    <div key={feature.name} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="capitalize">{feature.name}</span>
                        <span className="font-mono">{feature.value}%</span>
                      </div>
                      <Progress value={parseFloat(feature.value)} className="h-2" />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="flex flex-col items-center text-center">
                <Brain className="h-10 w-10 text-muted-foreground mb-3" />
                <h4 className="font-semibold mb-1">Per-Pair ML Training</h4>
                <p className="text-sm text-muted-foreground max-w-md">
                  Each pair trains its own ML model after {mlStatus?.min_trades_per_pair || 5} closed trades.
                  Pairs without enough data use technical analysis.
                </p>
                {mlStatus && (
                  <p className="text-xs text-muted-foreground mt-2">
                    {mlStatus.trained_pairs_count}/{mlStatus.total_pairs} pairs trained
                  </p>
                )}
              </div>
              
              {mlStatus?.pair_progress && (
                <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                  {mlStatus.pair_progress.map((pairStatus) => (
                    <div 
                      key={pairStatus.pair}
                      className={`p-3 rounded-lg border ${
                        pairStatus.is_trained 
                          ? 'border-success/50 bg-success/5' 
                          : 'border-border bg-muted/30'
                      }`}
                      data-testid={`ml-status-${pairStatus.pair.replace('/', '-')}`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-sm">{pairStatus.pair}</span>
                        {pairStatus.is_trained ? (
                          <Badge variant="default" className="text-xs">ML Active</Badge>
                        ) : (
                          <Badge variant="secondary" className="text-xs">Bootstrap</Badge>
                        )}
                      </div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">Progress</span>
                          <span className="font-mono">
                            {pairStatus.closed_trades}/{pairStatus.required_trades}
                          </span>
                        </div>
                        <Progress 
                          value={pairStatus.progress_percent} 
                          className="h-1.5" 
                        />
                        {pairStatus.is_trained && pairStatus.accuracy > 0 && (
                          <div className="flex justify-between text-xs pt-1">
                            <span className="text-muted-foreground">Accuracy</span>
                            <span className="font-mono text-success">
                              {(pairStatus.accuracy * 100).toFixed(0)}%
                            </span>
                          </div>
                        )}
                        {!pairStatus.is_trained && (
                          <p className="text-xs text-muted-foreground pt-1">
                            {pairStatus.required_trades - pairStatus.closed_trades} more trades needed
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
