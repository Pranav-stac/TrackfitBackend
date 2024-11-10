from django.contrib import admin
from .models import User, UserProfile, WeekGraph, DietTable, Exercise, ExerciseDetail, PersonalInfo

# Register your models here.
admin.site.register(User)
admin.site.register(UserProfile)
admin.site.register(WeekGraph)
admin.site.register(DietTable)
admin.site.register(Exercise)
admin.site.register(ExerciseDetail)
admin.site.register(PersonalInfo)
