from django.contrib.auth.models import AbstractUser
from django.db import models

# Create your models here.

class User(AbstractUser):
    email = models.EmailField(unique=True)

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    total_calories = models.IntegerField(default=0)
    total_steps = models.IntegerField(default=0)
    sleep_quality = models.CharField(max_length=100)
    overall_health = models.CharField(max_length=100)
    map_location = models.CharField(max_length=255)

class WeekGraph(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    day = models.CharField(max_length=10)
    calories = models.IntegerField()

class DietTable(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    day = models.CharField(max_length=10)
    dish1 = models.CharField(max_length=100)
    dish2 = models.CharField(max_length=100)
    dish3 = models.CharField(max_length=100)

class Exercise(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    day = models.CharField(max_length=10)

class ExerciseDetail(models.Model):
    exercise = models.ForeignKey(Exercise, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    sets = models.IntegerField()
    reps = models.IntegerField()
    description = models.TextField()
    duration = models.DurationField()
    gif = models.URLField()

class PersonalInfo(models.Model):
    age = models.IntegerField(null=True, blank=True)
    current_height = models.FloatField(help_text="Height in centimeters", null=True, blank=True)
    current_weight = models.FloatField(help_text="Weight in kilograms", null=True, blank=True)
    goal_weight = models.FloatField(help_text="Goal weight in kilograms", null=True, blank=True)
    fitness_goal = models.CharField(max_length=20, null=True, blank=True)
    diet_preference = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return f"{self.user_profile.user.username}'s Personal Info"
