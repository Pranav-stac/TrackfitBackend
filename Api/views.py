from django.http import HttpResponse
from rest_framework_simplejwt.tokens import RefreshToken
from django.shortcuts import render
import numpy as np
# Create your views here.
def index(request):
    return HttpResponse("Hello, world. You're at the Api index.")
from rest_framework import viewsets
from .models import User, UserProfile, WeekGraph, DietTable, Exercise, ExerciseDetail
from .serializers import UserSerializer, UserProfileSerializer, WeekGraphSerializer, DietTableSerializer, ExerciseSerializer, ExerciseDetailSerializer

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer

class WeekGraphViewSet(viewsets.ModelViewSet):
    queryset = WeekGraph.objects.all()
    serializer_class = WeekGraphSerializer

class DietTableViewSet(viewsets.ModelViewSet):
    queryset = DietTable.objects.all()
    serializer_class = DietTableSerializer

class ExerciseViewSet(viewsets.ModelViewSet):
    queryset = Exercise.objects.all()
    serializer_class = ExerciseSerializer

class ExerciseDetailViewSet(viewsets.ModelViewSet):
    queryset = ExerciseDetail.objects.all()
    serializer_class = ExerciseDetailSerializer

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .serializers import HomeDataSerializer
from .models import UserProfile, WeekGraph
from django.views.decorators.csrf import csrf_exempt
# Removed csrf_exempt decorator

class HomeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            # Attempt to get the user's profile
            user_profile = UserProfile.objects.get(user=request.user)
        except UserProfile.DoesNotExist:
            # Provide default values if the UserProfile does not exist
            user_profile = UserProfile(
                user=request.user,
                total_calories=0,
                total_steps=0,
                sleep_quality="Unknown",
                overall_health="Unknown",
                map_location="No map data available"
            )
            # Optionally, save the default profile to the database
            user_profile.save()

        # Get the week graph data
        week_graph_data = WeekGraph.objects.filter(user_profile=user_profile)
        week_graph = {entry.day: entry.calories for entry in week_graph_data}

        # Prepare the data
        data = {
            "total_calories": user_profile.total_calories,
            "total_steps": user_profile.total_steps,
            "sleep_quality": user_profile.sleep_quality,
            "week_graph": week_graph,
            "overall_health": user_profile.overall_health,
            "map": user_profile.map_location,
        }

        serializer = HomeDataSerializer(data)
        return Response(serializer.data,stt)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from .models import DietTable, UserProfile
from .serializers import DietTableSerializer
from django.contrib.auth import get_user_model

# Removed csrf_exempt decorator
class DietTableView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Get the user's profile
        try:
            user_profile = UserProfile.objects.get(user=request.user)
        except UserProfile.DoesNotExist:
            return Response({"error": "User profile not found."}, status=status.HTTP_404_NOT_FOUND)

        # Add the user profile to the request data
        request.data['user_profile'] = user_profile.id

        # Serialize the data
        serializer = DietTableSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

# Removed csrf_exempt decorator
@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
    User = get_user_model()  # Use the custom user model
    username = request.data.get('username')
    password = request.data.get('password')
    email = request.data.get('email')  # Assuming email is part of the request data

    if username is None or password is None or email is None:
        return Response({'error': 'Please provide username, password, and email'}, status=status.HTTP_400_BAD_REQUEST)

    if User.objects.filter(username=username).exists():
        return Response({'error': 'Username already exists'}, status=status.HTTP_400_BAD_REQUEST)

    if User.objects.filter(email=email).exists():
        return Response({'error': 'Email already exists'}, status=status.HTTP_400_BAD_REQUEST)

    user = User.objects.create_user(username=username, password=password, email=email)
    return Response({'message': 'User created successfully'}, status=status.HTTP_201_CREATED)

# Removed csrf_exempt decorator
@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    username = request.data.get('username')
    password = request.data.get('password')
    user = authenticate(username=username, password=password)
    if user is not None:
        login(request, user)
        refresh = RefreshToken.for_user(user)
        return Response({
            'message': 'Login successful',
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }, status=status.HTTP_200_OK)
    else:
        return Response({'error': 'Invalid credentials'}, status=status.HTTP_400_BAD_REQUEST)

from rest_framework import viewsets
from .models import PersonalInfo
from .serializers import PersonalInfoSerializer

class PersonalInfoViewSet(viewsets.ModelViewSet):
    queryset = PersonalInfo.objects.all()
    serializer_class = PersonalInfoSerializer

import cv2
import mediapipe as mp
import numpy as np
import os
import glob

def pose_comparison(pose_name):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360-angle
        return angle

    def get_pose_landmarks(image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            return landmarks
        return None

    def compare_poses(pose1, pose2):
        if pose1 is None or pose2 is None:
            return 0, {}

        angles_to_compare = [
            ("Left shoulder", 11, 13, 15),
            ("Right shoulder", 12, 14, 16),
            ("Left elbow", 13, 11, 23),
            ("Right elbow", 14, 12, 24),
            ("Neck", 7, 11, 12),
            ("Left hip", 23, 25, 27),
            ("Right hip", 24, 26, 28),
            ("Left knee", 25, 23, 27),
            ("Right knee", 26, 24, 28),
            ("Left ankle", 27, 25, 31),
            ("Right ankle", 28, 26, 32),
        ]

        angle_similarities = {}
        for name, *points in angles_to_compare:
            angle1 = calculate_angle(pose1[points[0]], pose1[points[1]], pose1[points[2]])
            angle2 = calculate_angle(pose2[points[0]], pose2[points[1]], pose2[points[2]])
            angle_diff = abs(angle1 - angle2)
            similarity = max(0, 1 - angle_diff / 90)
            angle_similarities[name] = similarity * 100

        overall_similarity = np.mean(list(angle_similarities.values()))
        return overall_similarity, angle_similarities

    def load_reference_pose(pose_name):
        dataset_path = r'dataset'
        pose_path = os.path.join(dataset_path, pose_name)
        if not os.path.exists(pose_path):
            print(f"Error: Folder for '{pose_name}' not found.")
            return None

        image_files = glob.glob(os.path.join(pose_path, '*'))
        if not image_files:
            print(f"Error: No images found in the folder for '{pose_name}'.")
            return None

        first_image_path = image_files[0]
        img = cv2.imread(first_image_path)
        if img is None:
            print(f"Error: Unable to read the first image in '{pose_name}' folder.")
            return None

        pose_landmarks = get_pose_landmarks(img)
        if pose_landmarks is None:
            print(f"Error: Unable to detect pose in the first image for '{pose_name}'.")
            return None

        print(f"Loaded reference pose for '{pose_name}' from {os.path.basename(first_image_path)}.")
        return pose_landmarks

    # Load the reference pose
    reference_pose = load_reference_pose(pose_name)

    if reference_pose is None:
        return

    # Start video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break

        # Get current pose
        current_pose = get_pose_landmarks(frame)

        # Compare poses
        similarity, angle_similarities = compare_poses(current_pose, reference_pose) if current_pose is not None else (0, {})

        # Draw pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display overall similarity percentage
        color = (0, 255, 0) if similarity > 50 else (0, 0, 255)
        cv2.putText(frame, f'Overall Similarity: {similarity:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display individual angle similarities
        y_offset = 60
        for name, sim in angle_similarities.items():
            color = (0, 255, 0) if sim > 50 else (0, 0, 255)
            cv2.putText(frame, f'{name}: {sim:.2f}%', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20

        # Display the frame
        cv2.imshow('Pose Comparison', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    exit(0)  # Exit the program after closing the OpenCV window

from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def upload_view(request):
    pose_comparison('vriksasana')
@csrf_exempt
def compare_pose(request):
    if request.method == 'POST':
        pose_name = request.POST.get('pose_name')
        pose_comparison(pose_name)


from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
import google.generativeai as genai  # Assuming this is the Gemini API client

import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import io



import io
import json  # Import the json module
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import google.generativeai as genai  # Assuming this is the Gemini API client

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from PIL import Image
import io
import base64
import google.generativeai as genai  # Assuming this is the Gemini API client

@csrf_exempt
def analyze_image(request):
    if request.method == 'POST':
        try:
            # Check if the request contains a file
            if 'image' in request.FILES:
                image_file = request.FILES['image']
                image = Image.open(image_file)

                # Save the image to a temporary file
                temp_image_file = io.BytesIO()
                image.save(temp_image_file, format='PNG')
                temp_image_file.seek(0)

                genai.configure(api_key='AIzaSyBtg_JyBr0sZQ4l_dVyvkl0KXZAfhFFr5E')
                myfile = genai.upload_file(temp_image_file)
                model = genai.GenerativeModel("gemini-1.5-flash")
                result = model.generate_content(
                    [myfile, "\n\n", "Give me the amount of calories per item that is present in the image"]
                )

                return JsonResponse({'message': 'Image processed successfully', 'result': result.text})
            else:
                return JsonResponse({'error': 'No image file provided'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


class TextProcessingView(APIView):
    def post(self, request, *args, **kwargs):
        # Get the text input from the request
        text_input = request.data.get('text')
        if not text_input:
            return JsonResponse({'error': 'No text input provided'}, status=400)

        # Process the text input using Gemini API
        try:
            # Configure the API key
            genai.configure(api_key='AIzaSyBtg_JyBr0sZQ4l_dVyvkl0KXZAfhFFr5E')
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            result = model.generate_content([text_input])
            return JsonResponse({'output': result.text}, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)