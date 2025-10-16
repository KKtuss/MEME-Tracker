# Quick Start Guide - Meme Tracker

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
py -m pip install -r requirements.txt
```

### Step 2: Test Installation
```bash
py test_installation_opencv.py
```

### Step 3: Run the Application
```bash
py main_app_opencv.py
```

## 🎯 How to Use

1. **Start Camera**: Click "Start Camera" button
2. **Assign Images**: 
   - Go to "Pose Configuration" tabs
   - Click "Browse" to select images for different poses
   - Assign images to poses like "Smile Left Hand", "Look Right", etc.
3. **Test Detection**: 
   - Smile and hold your left hand up → Shows assigned image
   - Look left/right → Shows assigned image
   - Try different combinations!

## 🎨 Supported Poses

### Smile Poses
- **Smile Left Hand**: Smile + left hand visible
- **Smile Right Hand**: Smile + right hand visible  
- **Smile Both Hands**: Smile + both hands visible
- **Smile No Hands**: Just smiling

### Neutral Poses
- **Neutral Left Hand**: Neutral face + left hand
- **Neutral Right Hand**: Neutral face + right hand
- **Neutral Both Hands**: Neutral face + both hands
- **Neutral No Hands**: Just neutral face

### Eye Direction
- **Look Left**: Looking to the left
- **Look Right**: Looking to the right
- **Look Center**: Looking straight ahead

## 💡 Tips for Best Results

1. **Lighting**: Use good, even lighting on your face
2. **Distance**: Stay 1-3 feet from the camera
3. **Background**: Use a plain background for better hand detection
4. **Pose Clearly**: Make clear, deliberate poses for better detection

## 🔧 Troubleshooting

- **Camera not working?** Close other applications using the camera
- **Poor detection?** Improve lighting and background
- **Images not loading?** Check file paths are correct

## 📁 File Structure

- `main_app_opencv.py` - Main application (use this one!)
- `camera_tracker_opencv.py` - Core detection logic
- `images/` - Put your assigned images here
- `pose_mappings.json` - Auto-saved pose assignments

Enjoy your meme tracker! 🎉

