// Get class names as array
import { fdc_ids_as_array } from "./constants.js";
import { getFoodData } from "./get_data.js";
import { get_all_food_data_from_supabase } from "./get_data.js";

console.log(fdc_ids_as_array)

// Get all food data in one hit from Supabase and save it to a constant
const data = await get_all_food_data_from_supabase();
console.log("Logging data:")
console.log(data);


// Check to see if TF.js is available
const tfjs_status = document.getElementById("tfjs_status");

if (tfjs_status) {
    tfjs_status.innerText = "Loaded TensorFlow.js - version:" + tf.version.tfjs;
}

// Setup the model code
let model; // This is in global scope

const loadModel = async () => {
    try {
        const tfliteModel = await tflite.loadTFLiteModel(
            "models/2022-01-16-nutrify_model_100_foods_manually_cleaned_10_classes_foods_v1.tflite"
        );
        model = tfliteModel; // assigning it to the global scope model as tfliteModel can only be used within this scope

        //  Check if model loaded
        if (tfliteModel) {
            model_status.innerText = "Model loaded";
        }
    } catch (error) {
        console.log(error);
    }
};

loadModel();

// Function to classify image
function classifyImage(model, image) {
    // Preprocess image
    image = tf.image.resizeBilinear(image, [240, 240]); // image size needs to be same as model inputs - EffNetB1 takes 240x240
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

    console.log("Output of model:");
    console.log(output.arraySync());
    console.log(output.arraySync()[0]); // arraySync() Returns an array to use

    console.log("After calling softmax on the output:");
    console.log(output_values.arraySync());

    // Update HTML
    const predicted_class_string = fdc_ids_as_array[output_values.argMax().arraySync()];
    predicted_class.textContent = predicted_class_string;
    // predicted_prob.textContent = output_values.max().arraySync() * 100 + "%";
    // Get data from Supabase and update HTML
    getFoodData(predicted_class_string, data);
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

            // Classify image (and update page with food info)
            var startTime = performance.now()
            classifyImage(model, currImage);
            var endTime = performance.now()
            document.getElementById("time_taken").textContent = `${(endTime - startTime) / 1000} seconds`
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
