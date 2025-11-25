import { useQuery, useMutation } from '@tanstack/react-query';
import { 
  DollarSign, TrendingUp, TrendingDown, Activity, 
  Power, Brain, Clock, Target, AlertCircle, Loader2,
  Wifi, WifiOff, AlertTriangle, X, CheckCircle, XCircle
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { useToast } from '@/hooks/use-toast';
import { authFetch } from '@/lib/auth';
import { queryClient } from '@/lib/queryClient';

interface ApiStatus {
  api_configured: boolean;
  api_connected: boolean;
  account_info: {
    balance: number;
    available: number;
    currency: string;
  } | null;
  missing_credentials: (string | null)[];
}

interface DashboardData {
  account_balance: number;
  available_capital: number;
  total_profit_loss: number;
  win_rate: number;
  ai_status: boolean;
  total_trades: number;
  open_trades: number;
  ml_model_active: boolean;
  ml_accuracy: number;
  trades_until_ml: number;
  trades_until_retrain: number;
}

interface Trade {
  id: number;
  pair: string;
  direction: string;
  entry_price: number;
  current_price: number | null;
  profit_loss: number;
  status: string;
  opened_at: string;
  strategy_used: string;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(value);
}

function formatNumber(value: number | string | null | undefined, decimals = 2): string {
  if (value === null || value === undefined) return '0.00';
  const num = typeof value === 'string' ? parseFloat(value) : value;
  if (isNaN(num)) return '0.00';
  return num.toFixed(decimals);
}

function StatCard({ 
  title, 
  value, 
  icon: Icon, 
  trend, 
  subtitle,
  loading 
}: { 
  title: string; 
  value: string; 
  icon: any; 
  trend?: 'up' | 'down' | 'neutral';
  subtitle?: string;
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
        {subtitle && (
          <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const { toast } = useToast();

  const { data: apiStatus, isLoading: apiStatusLoading } = useQuery<ApiStatus>({
    queryKey: ['/api/status/'],
    queryFn: async () => {
      const res = await fetch('/api/status/');
      if (!res.ok) throw new Error('Failed to fetch API status');
      return res.json();
    },
    refetchInterval: 5000,
  });

  const { data: dashboard, isLoading: dashboardLoading } = useQuery<DashboardData>({
    queryKey: ['/api/dashboard/'],
    queryFn: async () => {
      const res = await authFetch('/api/dashboard/');
      if (!res.ok) throw new Error('Failed to fetch dashboard');
      return res.json();
    },
    refetchInterval: 3000,
  });

  const { data: openTrades, isLoading: tradesLoading } = useQuery<Trade[]>({
    queryKey: ['/api/trades/open/'],
    queryFn: async () => {
      const res = await authFetch('/api/trades/open/');
      if (!res.ok) throw new Error('Failed to fetch trades');
      return res.json();
    },
    refetchInterval: 2000,
  });

  const liveTotalPnL = openTrades?.reduce((sum, trade) => {
    const pnl = typeof trade.profit_loss === 'string' 
      ? parseFloat(trade.profit_loss) 
      : (trade.profit_loss ?? 0);
    return sum + pnl;
  }, 0) ?? 0;

  const { data: recentTrades } = useQuery<Trade[]>({
    queryKey: ['/api/trades/history/', { limit: 10 }],
    queryFn: async () => {
      const res = await authFetch('/api/trades/history/?limit=10');
      if (!res.ok) throw new Error('Failed to fetch trade history');
      return res.json();
    },
    refetchInterval: 30000,
  });

  const toggleAIMutation = useMutation({
    mutationFn: async () => {
      const res = await authFetch('/api/settings/toggle-ai/', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.message || 'Failed to toggle AI');
      }
      return data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/api/dashboard/'] });
      queryClient.invalidateQueries({ queryKey: ['/api/status/'] });
      queryClient.invalidateQueries({ queryKey: ['/api/trades/open/'] });
      toast({
        title: data.ai_enabled ? 'AI Trading Started' : 'AI Trading Stopped',
        description: data.message,
      });
    },
    onError: (error: Error) => {
      toast({
        title: 'Cannot Start AI Trading',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const closePositionMutation = useMutation({
    mutationFn: async (tradeId: number) => {
      const res = await authFetch(`/api/trades/${tradeId}/close/`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Failed to close position');
      }
      return data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/api/trades/open/'] });
      queryClient.invalidateQueries({ queryKey: ['/api/trades/history/'] });
      queryClient.invalidateQueries({ queryKey: ['/api/dashboard/'] });
      toast({
        title: 'Position Closed',
        description: `${data.pair}: ${formatCurrency(data.profit_loss)} (${data.outcome})`,
        variant: data.profit_loss >= 0 ? 'default' : 'destructive',
      });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to Close Position',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const closeProfitableMutation = useMutation({
    mutationFn: async () => {
      const res = await authFetch('/api/trades/close-profitable/', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Failed to close profitable trades');
      }
      return data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/api/trades/open/'] });
      queryClient.invalidateQueries({ queryKey: ['/api/trades/history/'] });
      queryClient.invalidateQueries({ queryKey: ['/api/dashboard/'] });
      toast({
        title: 'Profitable Trades Closed',
        description: `Closed ${data.closed} trades for ${formatCurrency(data.total_profit)} profit`,
      });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to Close Profitable Trades',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const closeAllMutation = useMutation({
    mutationFn: async () => {
      const res = await authFetch('/api/trades/close-all/', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Failed to close all positions');
      }
      return data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/api/trades/open/'] });
      queryClient.invalidateQueries({ queryKey: ['/api/trades/history/'] });
      queryClient.invalidateQueries({ queryKey: ['/api/dashboard/'] });
      toast({
        title: 'All Positions Closed',
        description: `Closed ${data.closed} positions. Net P/L: ${formatCurrency(data.total_pnl)}`,
      });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to Close Positions',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const apiConfigured = apiStatus?.api_configured ?? false;
  const apiConnected = apiStatus?.api_connected ?? false;

  const aiEnabled = dashboard?.ai_status ?? false;
  const mlProgress = dashboard ? Math.min(100, ((30 - dashboard.trades_until_ml) / 30) * 100) : 0;

  return (
    <div className="p-6 space-y-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold">Trading Dashboard</h1>
        <p className="text-muted-foreground">Monitor your AI trading performance in real-time</p>
      </div>

      {!apiStatusLoading && !apiConfigured && (
        <Alert variant="destructive" data-testid="alert-api-not-configured">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>API Credentials Required</AlertTitle>
          <AlertDescription>
            Capital.com API credentials are not configured. Please add the following secrets to enable trading:
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li>CAPITAL_COM_API_KEY</li>
              <li>CAPITAL_COM_PASSWORD</li>
              <li>CAPITAL_COM_IDENTIFIER</li>
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {!apiStatusLoading && apiConfigured && !apiConnected && (
        <Alert variant="destructive" data-testid="alert-api-not-connected">
          <WifiOff className="h-4 w-4" />
          <AlertTitle>API Connection Failed</AlertTitle>
          <AlertDescription>
            Unable to connect to Capital.com API. Please verify your credentials are correct.
          </AlertDescription>
        </Alert>
      )}

      {!apiStatusLoading && apiConnected && (
        <Alert data-testid="alert-api-connected" className="border-success/50 bg-success/10">
          <Wifi className="h-4 w-4 text-success" />
          <AlertTitle className="text-success">Connected to Capital.com</AlertTitle>
          <AlertDescription>
            API connection established. Ready to trade on demo account.
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Account Balance"
          value={formatCurrency(dashboard?.account_balance ?? 0)}
          icon={DollarSign}
          loading={dashboardLoading}
        />
        <StatCard
          title="Available Capital"
          value={formatCurrency(dashboard?.available_capital ?? 0)}
          icon={Target}
          loading={dashboardLoading}
        />
        <StatCard
          title="Open P/L"
          value={formatCurrency(liveTotalPnL)}
          icon={Activity}
          trend={liveTotalPnL >= 0 ? 'up' : 'down'}
          subtitle={`${openTrades?.length ?? 0} open positions`}
          loading={tradesLoading}
        />
        <StatCard
          title="Win Rate"
          value={`${formatNumber(dashboard?.win_rate ?? 0)}%`}
          icon={TrendingUp}
          trend={(dashboard?.win_rate ?? 0) >= 50 ? 'up' : 'down'}
          loading={dashboardLoading}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Power className="h-5 w-5" />
              AI Control Center
            </CardTitle>
            <CardDescription>
              Control your autonomous trading engine
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`h-3 w-3 rounded-full ${aiEnabled ? 'bg-success animate-pulse' : 'bg-muted'}`} />
                <span className="font-medium">{aiEnabled ? 'Active' : 'Inactive'}</span>
              </div>
              <Badge variant={aiEnabled ? 'default' : 'secondary'}>
                {aiEnabled ? 'Running' : 'Stopped'}
              </Badge>
            </div>

            <Button
              className="w-full"
              size="lg"
              variant={aiEnabled ? 'destructive' : 'default'}
              onClick={() => toggleAIMutation.mutate()}
              disabled={toggleAIMutation.isPending || (!aiEnabled && !apiConfigured)}
              data-testid="button-toggle-ai"
            >
              {toggleAIMutation.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Power className="mr-2 h-4 w-4" />
              )}
              {aiEnabled ? 'Stop AI Trading' : 'Start AI Trading'}
            </Button>
            
            {!apiConfigured && !aiEnabled && (
              <p className="text-xs text-muted-foreground text-center">
                Configure API credentials to enable AI trading
              </p>
            )}

            <div className="space-y-3 pt-4 border-t">
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm">ML Model Status</span>
              </div>
              
              {dashboard?.ml_model_active ? (
                <div className="space-y-2">
                  <Badge variant="default" className="bg-success">
                    ML Active
                  </Badge>
                  <p className="text-sm text-muted-foreground">
                    Accuracy: <span className="font-mono font-medium">{dashboard.ml_accuracy}%</span>
                  </p>
                  {dashboard.trades_until_retrain > 0 && (
                    <p className="text-xs text-muted-foreground">
                      Next retrain in {dashboard.trades_until_retrain} trades
                    </p>
                  )}
                </div>
              ) : (
                <div className="space-y-2">
                  <Badge variant="secondary">Technical Analysis Only</Badge>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span>Training progress</span>
                      <span className="font-mono">{30 - (dashboard?.trades_until_ml ?? 30)}/30</span>
                    </div>
                    <Progress value={mlProgress} className="h-2" />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {dashboard?.trades_until_ml ?? 30} trades until ML activation
                  </p>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between pt-4 border-t text-sm">
              <span className="text-muted-foreground">Total Trades</span>
              <span className="font-mono font-medium">{dashboard?.total_trades ?? 0}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Open Positions</span>
              <span className="font-mono font-medium">{dashboard?.open_trades ?? 0}</span>
            </div>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Active Trades
                {openTrades && openTrades.length > 0 && (
                  <Badge variant="secondary" className="ml-2 font-mono">
                    {openTrades.length}
                  </Badge>
                )}
              </CardTitle>
              <CardDescription>
                Currently open positions
              </CardDescription>
            </div>
            {openTrades && openTrades.length > 0 && (
              <div className="flex gap-2">
                <Button
                  size="sm"
                  className="bg-blue-600 hover:bg-blue-700 text-white border-blue-600"
                  onClick={() => closeProfitableMutation.mutate()}
                  disabled={closeProfitableMutation.isPending || !openTrades.some(t => t.profit_loss > 0)}
                  data-testid="button-take-profit"
                >
                  {closeProfitableMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-1" />
                  ) : (
                    <CheckCircle className="h-4 w-4 mr-1" />
                  )}
                  Take Profit
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => closeAllMutation.mutate()}
                  disabled={closeAllMutation.isPending}
                  data-testid="button-close-all"
                >
                  {closeAllMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-1" />
                  ) : (
                    <XCircle className="h-4 w-4 mr-1" />
                  )}
                  Close All
                </Button>
              </div>
            )}
          </CardHeader>
          <CardContent>
            {tradesLoading ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : openTrades && openTrades.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Pair</TableHead>
                    <TableHead>Direction</TableHead>
                    <TableHead className="text-right">Entry</TableHead>
                    <TableHead className="text-right">Current</TableHead>
                    <TableHead className="text-right">P/L</TableHead>
                    <TableHead className="text-right">Action</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {openTrades.map((trade) => {
                    const profitLoss = typeof trade.profit_loss === 'string' 
                      ? parseFloat(trade.profit_loss) 
                      : (trade.profit_loss ?? 0);
                    const isProfit = profitLoss > 0;
                    const isLoss = profitLoss < 0;
                    
                    return (
                      <TableRow key={trade.id} data-testid={`row-trade-${trade.id}`}>
                        <TableCell className="font-medium">{trade.pair}</TableCell>
                        <TableCell>
                          <Badge 
                            variant={trade.direction === 'buy' ? 'default' : 'destructive'}
                            className={trade.direction === 'buy' ? 'bg-blue-600 hover:bg-blue-700' : ''}
                          >
                            {trade.direction.toUpperCase()}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {formatNumber(trade.entry_price, 5)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {trade.current_price ? formatNumber(trade.current_price, 5) : formatNumber(trade.entry_price, 5)}
                        </TableCell>
                        <TableCell className={`text-right font-mono font-semibold ${
                          isProfit ? 'text-green-500' : isLoss ? 'text-red-500' : ''
                        }`}>
                          {isProfit ? '+' : ''}{formatCurrency(profitLoss)}
                        </TableCell>
                        <TableCell className="text-right">
                          <Button
                            size="icon"
                            variant="destructive"
                            onClick={() => closePositionMutation.mutate(trade.id)}
                            disabled={closePositionMutation.isPending}
                            data-testid={`button-close-trade-${trade.id}`}
                          >
                            <X className="h-4 w-4 text-white" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            ) : (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <AlertCircle className="h-8 w-8 text-muted-foreground mb-2" />
                <p className="text-muted-foreground">No active trades</p>
                <p className="text-sm text-muted-foreground">
                  {aiEnabled ? 'Waiting for trading signals...' : 'Start AI to begin trading'}
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Recent Trades
          </CardTitle>
          <CardDescription>
            Your last 10 closed positions
          </CardDescription>
        </CardHeader>
        <CardContent>
          {recentTrades && recentTrades.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Pair</TableHead>
                  <TableHead>Direction</TableHead>
                  <TableHead>Strategy</TableHead>
                  <TableHead className="text-right">Entry</TableHead>
                  <TableHead className="text-right">Exit</TableHead>
                  <TableHead className="text-right">P/L</TableHead>
                  <TableHead>Outcome</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recentTrades.map((trade) => {
                  const profitLoss = typeof trade.profit_loss === 'string' 
                    ? parseFloat(trade.profit_loss) 
                    : (trade.profit_loss ?? 0);
                  const isProfit = profitLoss > 0;
                  const isLoss = profitLoss < 0;
                  
                  return (
                    <TableRow key={trade.id} data-testid={`row-history-${trade.id}`}>
                      <TableCell className="font-medium">{trade.pair}</TableCell>
                      <TableCell>
                        <Badge 
                          variant={trade.direction === 'buy' ? 'default' : 'destructive'}
                          className={trade.direction === 'buy' ? 'bg-blue-600 hover:bg-blue-700' : ''}
                        >
                          {trade.direction.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {trade.strategy_used || 'Technical'}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {formatNumber(trade.entry_price, 5)}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {trade.current_price ? formatNumber(trade.current_price, 5) : '-'}
                      </TableCell>
                      <TableCell className={`text-right font-mono font-semibold ${
                        isProfit ? 'text-green-500' : isLoss ? 'text-red-500' : ''
                      }`}>
                        {isProfit ? '+' : ''}{formatCurrency(profitLoss)}
                      </TableCell>
                      <TableCell>
                        <Badge variant={isProfit ? 'default' : 'destructive'} className={isProfit ? 'bg-green-600' : ''}>
                          {isProfit ? 'WIN' : 'LOSS'}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <Clock className="h-8 w-8 text-muted-foreground mb-2" />
              <p className="text-muted-foreground">No trade history yet</p>
              <p className="text-sm text-muted-foreground">
                Completed trades will appear here
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
