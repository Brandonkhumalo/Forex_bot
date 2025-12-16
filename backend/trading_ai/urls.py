from django.contrib import admin
from django.urls import path, include, re_path
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static
from django.http import FileResponse
import os

def serve_react(request, path=''):
    """Serve React frontend for all non-API routes"""
    index_path = os.path.join(settings.BASE_DIR, 'static', 'index.html')
    if os.path.exists(index_path):
        return FileResponse(open(index_path, 'rb'), content_type='text/html')
    from django.http import HttpResponse
    return HttpResponse("Frontend not built. Run: npm run build", status=404)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
]

# Serve React frontend for all other routes
urlpatterns += [
    re_path(r'^(?!api/|admin/|static/).*$', serve_react),
]

# Serve static files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
