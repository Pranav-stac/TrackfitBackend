from django.urls import path, include
from . import views 
from rest_framework.routers import DefaultRouter
from .views import UserViewSet, UserProfileViewSet, WeekGraphViewSet, DietTableViewSet, ExerciseViewSet, ExerciseDetailViewSet, signup, login_view, PersonalInfoViewSet, TextProcessingView
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)


router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'user-profiles', UserProfileViewSet)
router.register(r'week-graphs', WeekGraphViewSet)
router.register(r'diet-tables', DietTableViewSet)
router.register(r'exercises', ExerciseViewSet)
router.register(r'exercise-details', ExerciseDetailViewSet)
router.register(r'personal-info', PersonalInfoViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('home/', views.HomeView.as_view(), name='home'),
    path('diet/', views.DietTableView.as_view(), name='diet_table'),
    path('signup/', views.signup, name='signup'),
    path('upload/', views.upload_view, name='upload'),
    path('login/', views.login_view, name='login'),
    path('compare_pose/', views.compare_pose, name='compare_pose'),
    path('analyze_image/', views.analyze_image, name='analyze_image'),
    path('process-text/', TextProcessingView.as_view(), name='process-text'),
]
