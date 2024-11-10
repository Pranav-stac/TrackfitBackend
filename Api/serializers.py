from rest_framework import serializers
from .models import User, UserProfile, WeekGraph, DietTable, Exercise, ExerciseDetail, PersonalInfo

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = '__all__'

class WeekGraphSerializer(serializers.ModelSerializer):
    class Meta:
        model = WeekGraph
        fields = '__all__'

class DietTableSerializer(serializers.ModelSerializer):
    class Meta:
        model = DietTable
        fields = '__all__'

class ExerciseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Exercise
        fields = '__all__'

class ExerciseDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExerciseDetail
        fields = '__all__'

class HomeDataSerializer(serializers.Serializer):
    total_calories = serializers.IntegerField()
    total_steps = serializers.IntegerField()
    sleep_quality = serializers.CharField()
    week_graph = serializers.DictField(child=serializers.IntegerField())
    overall_health = serializers.CharField()
    map = serializers.CharField()

class PersonalInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = PersonalInfo
        fields = '__all__'