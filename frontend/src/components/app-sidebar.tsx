import { useLocation, Link } from 'wouter';
import { 
  LayoutDashboard, BarChart3, Settings, LogOut, 
  TrendingUp, Moon, Sun, Brain
} from 'lucide-react';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
} from '@/components/ui/sidebar';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useAuth } from '@/lib/auth';
import { useTheme } from '@/lib/theme';
import { useQuery } from '@tanstack/react-query';
import { authFetch } from '@/lib/auth';

interface DashboardData {
  ai_status: boolean;
  account_balance: number;
  ml_model_active: boolean;
}

const navigationItems = [
  {
    title: 'Dashboard',
    icon: LayoutDashboard,
    href: '/dashboard',
  },
  {
    title: 'Analytics',
    icon: BarChart3,
    href: '/analytics',
  },
];

export function AppSidebar() {
  const [location] = useLocation();
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();

  const { data: dashboard } = useQuery<DashboardData>({
    queryKey: ['/api/dashboard/'],
    queryFn: async () => {
      const res = await authFetch('/api/dashboard/');
      if (!res.ok) throw new Error('Failed to fetch dashboard');
      return res.json();
    },
    refetchInterval: 10000,
  });

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  return (
    <Sidebar>
      <SidebarHeader className="p-4">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-primary">
            <TrendingUp className="h-5 w-5 text-primary-foreground" />
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-lg">TradingAI</span>
            <span className="text-xs text-muted-foreground">ML-Powered Trading</span>
          </div>
        </div>
      </SidebarHeader>

      <SidebarSeparator />

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navigationItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton 
                    asChild 
                    isActive={location === item.href}
                    data-testid={`nav-${item.title.toLowerCase()}`}
                  >
                    <Link href={item.href}>
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarSeparator />

        <SidebarGroup>
          <SidebarGroupLabel>AI Status</SidebarGroupLabel>
          <SidebarGroupContent>
            <div className="px-2 py-3 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">Trading Engine</span>
                <Badge variant={dashboard?.ai_status ? 'default' : 'secondary'}>
                  <div className={`h-2 w-2 rounded-full mr-1.5 ${dashboard?.ai_status ? 'bg-success animate-pulse' : 'bg-muted-foreground'}`} />
                  {dashboard?.ai_status ? 'Active' : 'Inactive'}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">ML Model</span>
                <Badge variant={dashboard?.ml_model_active ? 'default' : 'secondary'}>
                  <Brain className="h-3 w-3 mr-1" />
                  {dashboard?.ml_model_active ? 'Active' : 'Training'}
                </Badge>
              </div>
            </div>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarSeparator />

        <SidebarGroup>
          <SidebarGroupLabel>Account</SidebarGroupLabel>
          <SidebarGroupContent>
            <div className="px-2 py-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Balance</span>
                <span className="font-mono font-semibold">
                  {formatCurrency(dashboard?.account_balance ?? 0)}
                </span>
              </div>
            </div>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="p-4 space-y-2">
        <SidebarSeparator />
        
        <div className="flex items-center justify-between pt-2">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="text-sm font-medium">
                {user?.username?.charAt(0).toUpperCase() ?? 'U'}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-medium truncate max-w-24">
                {user?.username ?? 'User'}
              </span>
              <span className="text-xs text-muted-foreground truncate max-w-24">
                {user?.email ?? ''}
              </span>
            </div>
          </div>
          
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              data-testid="button-theme-toggle"
            >
              {theme === 'dark' ? (
                <Sun className="h-4 w-4" />
              ) : (
                <Moon className="h-4 w-4" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={logout}
              data-testid="button-logout"
            >
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
