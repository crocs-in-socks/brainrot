from django.contrib import admin
from django.urls import path, include
from frontend.views import landing_page

urlpatterns = [
    path('', landing_page, name='landing_page'),
    path('lstm/', include('lstm_api.urls')),
    path('admin/', admin.site.urls),
]
