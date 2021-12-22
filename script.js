const classes = {0: 'Apple',
1: 'Artichoke',
2: 'BBQ sauce',
3: 'Bacon',
4: 'Bagel',
5: 'Banana',
6: 'Beef',
7: 'Beer',
8: 'Blueberries',
9: 'Bread',
10: 'Broccoli',
11: 'Butter',
12: 'Cabbage',
13: 'Candy',
14: 'Cantaloupe',
15: 'Carrot',
16: 'Cheese',
17: 'Chicken',
18: 'Chicken wings',
19: 'Cocktail',
20: 'Coconut',
21: 'Coffee',
22: 'Cookie',
23: 'Corn chips',
24: 'Cream',
25: 'Cucumber',
26: 'Doughnut',
27: 'Egg',
28: 'Fish',
29: 'Fries',
30: 'Grape',
31: 'Guacamole',
32: 'Hamburger',
33: 'Honey',
34: 'Ice cream',
35: 'Lemon',
36: 'Lime',
37: 'Lobster',
38: 'Mango',
39: 'Milk',
40: 'Muffin',
41: 'Mushroom',
42: 'Olive oil',
43: 'Olives',
44: 'Onion',
45: 'Orange',
46: 'Orange juice',
47: 'Pancake',
48: 'Pasta',
49: 'Pastry',
50: 'Pear',
51: 'Pepper',
52: 'Pineapple',
53: 'Pizza',
54: 'Pomegranate',
55: 'Popcorn',
56: 'Potato',
57: 'Prawns',
58: 'Pretzel',
59: 'Pumpkin',
60: 'Radish',
61: 'Rice',
62: 'Salad',
63: 'Salt',
64: 'Sandwich',
65: 'Sausages',
66: 'Soft drink',
67: 'Spinach',
68: 'Squid',
69: 'Strawberries',
70: 'Sushi',
71: 'Tea',
72: 'Tomato',
73: 'Tomato sauce',
74: 'Waffle',
75: 'Watermelon',
76: 'Wine',
77: 'Zucchini'};

  // Check to see if TF.js is available
  const tfjs_status = document.getElementById("tfjs_status");

  if (tfjs_status) {
    tfjs_status.innerText = "Loaded TensorFlow.js - version:" + tf.version.tfjs;
  }

  let model; // This is in global scope

  const loadModel = async () => {
    try {
      const tfliteModel = await tflite.loadTFLiteModel(
        "models/nutrify_model_78_foods_v0.tflite"
      );
      model = tfliteModel; // assigning it to the global scope model as tfliteModel can only be used within this scope
      // console.log(tfliteModel);

      //  Check if model loaded
      if (tfliteModel) {
        model_status.innerText = "Model loaded";
      }
    } catch (error) {
      console.log(error);
    }

    // // Prepare input tensors.
    // const img = tf.browser.fromPixels(document.querySelector('img'));
    // const input = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);

    // // Run inference and get output tensors.
    // let outputTensor = tfliteModel.predict(input);
    // console.log(outputTensor.dataSync());
  };
  loadModel();

  // Function to classify image
  function classifyImage(model, image) {
    // Preprocess image
    image = tf.image.resizeBilinear(image, [224, 224]); // image size needs to be same as model inputs
    image = tf.expandDims(image);
    console.log(image);
    // console.log(model);

    // console.log(tflite.getDTypeFromTFLiteType("uint8")); // Gives int32 as output thus we cast int32 in below line
    // console.log(tflite.getDTypeFromTFLiteType("uint8"));
    console.log("converting image to different datatype...");
    image = tf.cast(image, "int32"); // Model requires uint8
    console.log("model about to predict...");
    const output = model.predict(image);
    const output_values = tf.softmax(output.arraySync()[0]);
    console.log("Arg max:");
    // console.log(output);
    console.log(output_values.arraySync());
    console.log("Output:");
    console.log(output.arraySync());
    console.log(output.arraySync()[0]); // arraySync() Returns an array to use

    // Update HTML
    predicted_class.textContent = classes[output_values.argMax().arraySync()];
    predicted_prob.textContent = output_values.max().arraySync() * 100 + "%";
  }

  // Image uploading
  const fileInput = document.getElementById("file-input");
  const image = document.getElementById("image");
  const uploadButton = document.getElementById("upload-button");

  function getImage() {
    if (!fileInput.files[0]) throw new Error("Image not found");
    const file = fileInput.files[0];

    // Get the data url from the image
    const reader = new FileReader();

    // When reader is ready display image
    reader.onload = function (event) {
      // Get data URL
      const dataUrl = event.target.result;

      // Create image object
      const imageElement = new Image();
      imageElement.src = dataUrl;

      // When image object loaded
      imageElement.onload = function () {
        // Display image
        image.setAttribute("src", this.src);

        // Log image parameters
        const currImage = tf.browser.fromPixels(imageElement);

        // Classify image
        classifyImage(model, currImage);
      };

      document.body.classList.add("image-loaded");
    };

    // Get data url
    reader.readAsDataURL(file);
  }

  // Add listener to see if someone uploads an image
  fileInput.addEventListener("change", getImage);
  uploadButton.addEventListener("click", () => fileInput.click());

  // console.log(tf.browser.fromPixels(fileInput.files[0]).print());

  // console.log(tf.browser.fromPixels(document.querySelector("image")));

  // const test_image = new ImageData(1, 1);
  // test_image.data[0] = 100;
  // test_image.data[1] = 150;
  // test_image.data[2] = 200;
  // test_image.data[3] = 255;

  // tf.browser.fromPixels(test_image).print();
