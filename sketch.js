// IA Peronista - Reconocimiento de Gestos
// Aplicación que utiliza IA para reconocer gestos peronistas

let classifier;
let handPose;
let video;
let hands = [];
let classification = "";
let loadingProgress = 0;
let loadingSteps = 3; // Number of loading steps
let currentStep = 0;
let loadingMessages = [
    {text: "🔍 Inicializando cámara...", progress: 15},
    {text: "🤖 Cargando modelo de IA...", progress: 40},
    {text: "✋ Preparando detección de gestos...", progress: 65},
    {text: "🎯 Optimizando precisión...", progress: 85},
    {text: "🚀 ¡Casi listo!...", progress: 95}
];

let loadingScreen;
let loadingText;
let loadingProgressBar;
let appContainer;
let sharePanel;
let isHandPoseReady = false;
let lastDetectionTime = 0;
const DETECTION_DELAY = 3000; // 3 seconds

function preload() {
  // Initialize UI elements
  loadingScreen = select('#loadingScreen');
  loadingText = select('.loading-text');
  loadingProgressBar = select('#loadingProgress');
  appContainer = select('#app');
  
  // Show initial loading message
  updateLoading('Iniciando IA Peronista...');
  
  // Initialize handPose model with explicit backend configuration
  handPose = ml5.handPose({
    flipHorizontal: true,
    runtime: 'tfjs',
    modelType: 'lite',
    maxHands: 2
  }, () => {
    console.log('HandPose model loaded!');
    isHandPoseReady = true;
    updateLoading(loadingMessages[currentStep++]);
    updateProgress(33);
  });
}

function setup() {
  // Set canvas size with a maximum width of 640px
  const maxWidth = 640;
  const aspectRatio = 4/3; // Standard 4:3 aspect ratio
  const canvasWidth = Math.min(windowWidth, maxWidth);
  const canvasHeight = canvasWidth / aspectRatio;
  
  // Create canvas with fixed aspect ratio
  let canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent('app');
  
  // Update loading progress
  updateLoading(loadingMessages[currentStep++]);
  updateProgress(66);

  // Create the webcam video with matching aspect ratio
  video = createCapture(VIDEO, { 
    video: {
      width: { ideal: maxWidth },
      height: { ideal: maxWidth / aspectRatio },
      aspectRatio: aspectRatio,
      facingMode: "user"
    },
    audio: false
  });
  
  video.elt.style.display = 'none'; // Hide the video element (we'll draw it manually)
  
  // Configure TensorFlow.js backend
  // This will automatically fall back to CPU if WebGL is not available
  if (typeof tf !== 'undefined') {
    tf.setBackend('cpu').then(() => {
      console.log('TensorFlow.js backend set to:', tf.getBackend());
    }).catch(err => {
      console.warn('Error setting TensorFlow.js backend:', err);
    });
  }

  // Set up the neural network
  let classifierOptions = {
    task: "classification",
  };
  classifier = ml5.neuralNetwork(classifierOptions);

  const modelDetails = {
    model: "model/model.json",
    metadata: "model/model_meta.json",
    weights: "model/model.weights.bin",
  };

  // Load the model
  classifier.load(modelDetails, () => {
    updateLoading(loadingMessages[currentStep++]);
    updateProgress(100);
    modelLoaded();
  });

  // Start the handPose detection once the model is ready
  if (isHandPoseReady) {
    handPose.detectStart(video, gotHands);
  } else {
    // If handPose isn't ready yet, check again shortly
    setTimeout(() => {
      if (handPose && handPose.detectStart) {
        handPose.detectStart(video, gotHands);
      }
    }, 500);
  }
}

function draw() {
  // Save the current drawing state
  push();
  
  // Mirror the entire drawing context horizontally
  translate(width, 0);
  scale(-1, 1);
  
  // Display the webcam video (already mirrored)
  image(video, 0, 0, width, height);
  
  // Draw the handPose keypoints
  if (hands[0]) {
    let hand = hands[0];
    
    // Draw hand skeleton
    drawHandSkeleton(hand);
    
    // If the model is loaded, make a classification
    if (isModelLoaded) {
      let inputData = flattenHandData();
      classifier.classify(inputData, gotClassification);
    }
  }
  
  // Restore the original drawing state
  pop();
  
  // Draw UI elements that should not be mirrored
  drawUI();
}

function drawHandSkeleton(hand) {
  if (!hand || !hand.keypoints) return;
  
  // Draw keypoints
  for (let i = 0; i < hand.keypoints.length; i++) {
    let keypoint = hand.keypoints[i];
    fill(255, 0, 0, 150);
    noStroke();
    circle(width - keypoint.x, keypoint.y, 8);
  }
  
  // Draw connections between keypoints if available
  if (handPose.HAND_CONNECTIONS) {
    for (let i = 0; i < handPose.HAND_CONNECTIONS.length; i++) {
      let startIndex = handPose.HAND_CONNECTIONS[i][0];
      let endIndex = handPose.HAND_CONNECTIONS[i][1];
      
      if (hand.keypoints[startIndex] && hand.keypoints[endIndex]) {
        let startX = width - hand.keypoints[startIndex].x;
        let startY = hand.keypoints[startIndex].y;
        let endX = width - hand.keypoints[endIndex].x;
        let endY = hand.keypoints[endIndex].y;
        
        stroke(0, 255, 0, 150);
        strokeWeight(2);
        line(startX, startY, endX, endY);
      }
    }
  }
}

function drawUI() {
  // No need to draw text anymore as we're using HTML elements
}

// convert the handPose data to a 1D array
function flattenHandData() {
  let hand = hands[0];
  let handData = [];
  for (let i = 0; i < hand.keypoints.length; i++) {
    let keypoint = hand.keypoints[i];
    handData.push(keypoint.x);
    handData.push(keypoint.y);
  }
  return handData;
}

// Callback function for when handPose outputs data
function gotHands(results) {
  hands = results;
}

// Callback function for when the classifier makes a classification
function gotClassification(results) {
  const newClassification = results[0].label;
  
  // Only update if classification changed
  if (newClassification !== classification) {
    classification = newClassification;
    
    // Show/hide share panel based on classification
    if (classification === "Peronista") {
      showSharePanel();
    } else {
      hideSharePanel();
    }
  }
}

// Show the share panel with animation
function showSharePanel() {
  if (!sharePanel) {
    sharePanel = select('#share-panel');
  }
  sharePanel.addClass('show');
  lastDetectionTime = millis();
}

// Hide the share panel with animation
function hideSharePanel() {
  if (sharePanel) {
    sharePanel.removeClass('show');
  }
}

// Share on social media
function shareOnSocial(platform) {
  const shareText = '¡Acabo de ser detectado como Peronista por la IA Peronista! 👋 #IAPeronista';
  const shareUrl = encodeURIComponent(window.location.href);
  const text = encodeURIComponent(shareText);
  
  let shareLink = '';
  
  switch(platform) {
    case 'facebook':
      shareLink = `https://www.facebook.com/sharer/sharer.php?u=${shareUrl}&quote=${text}`;
      break;
    case 'twitter':
      shareLink = `https://twitter.com/intent/tweet?text=${text}&url=${shareUrl}`;
      break;
    case 'whatsapp':
      shareLink = `https://wa.me/?text=${text} ${shareUrl}`;
      break;
  }
  
  if (shareLink) {
    window.open(shareLink, '_blank', 'width=600,height=400');
  }
}

// Update loading UI with smooth transitions
function updateLoading(message) {
  if (loadingText) {
    // Add fade out effect
    loadingText.style('opacity', '0');
    
    // After fade out, update text and fade in
    setTimeout(() => {
      loadingText.html(message.text || message);
      loadingText.style('opacity', '1');
      
      // Update progress if message includes progress info
      if (message.progress) {
        updateProgress(message.progress);
      }
    }, 200);
  }
}

// Update loading progress bar with smooth animation
function updateProgress(percent) {
  if (loadingProgressBar) {
    // Smooth transition for progress bar
    loadingProgressBar.style('transition', 'width 0.7s ease-out');
    loadingProgressBar.style('width', percent + '%');
    
    // Change color based on progress
    if (percent < 30) {
      loadingProgressBar.style('background', 'linear-gradient(90deg, #1e3c72, #2a5298)');
    } else if (percent < 70) {
      loadingProgressBar.style('background', 'linear-gradient(90deg, #2a5298, #7db9e8)');
    } else {
      loadingProgressBar.style('background', 'linear-gradient(90deg, #4CAF50, #8BC34A)');
    }
    
    loadingProgress = percent;
  }
}

// Callback function for when the pre-trained model is loaded
function modelLoaded() {
  isModelLoaded = true;
  
  // Hide loading screen with a delay for better UX
  setTimeout(() => {
    loadingScreen.addClass('fade-out');
    appContainer.style('display', 'block');
    
    // Remove loading screen from DOM after animation
    setTimeout(() => {
      loadingScreen.remove();
    }, 500);
  }, 1000);
}
