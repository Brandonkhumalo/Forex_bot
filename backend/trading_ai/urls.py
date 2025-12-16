from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.http import FileResponse, HttpResponse
from django.views.static import serve as static_serve
import os

def serve_react(request, path=''):
    """Serve React frontend for all non-API routes"""
    index_path = os.path.join(settings.BASE_DIR, 'static', 'index.html')
    if os.path.exists(index_path):
        return FileResponse(open(index_path, 'rb'), content_type='text/html')
    return HttpResponse("Frontend not built. Run: npm run build", status=404)

def serve_assets(request, path):
    """Serve React assets from /assets/ path"""
    assets_path = os.path.join(settings.BASE_DIR, 'static', 'assets')
    return static_serve(request, path, document_root=assets_path)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
    re_path(r'^assets/(?P<path>.*)$', serve_assets),
]

# Serve React frontend for all other routes
urlpatterns += [
    re_path(r'^(?!api/|admin/|static/|assets/).*$', serve_react),
]
