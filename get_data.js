import { fdc_ids_string_index } from "./constants.js";
import { supabase } from "./supabaseClient.js";

// Get all data in one hit from Supabase (this is all rows in the DB)
export async function get_all_food_data_from_supabase() {
    let { data, error } = await supabase
        .from("food_data_central_nutrition_data_test")
        .select("*")

    if (error) {
        console.log(error);
        return;
    }
    console.log(data);
    return data;
};

// Make a function to get food data and then update the DOM elements
export async function getFoodData(food_selection, data) {
    // Start timer
    // var startTime = performance.now()

    console.log("Logging data again...")
    console.log(data);
    console.log(fdc_ids_string_index);
    const target_food_code = fdc_ids_string_index[food_selection.toLowerCase()];
    const target_food_data = data.find(element => element["fdc_id"] == target_food_code)
    // const data = await get_food_data_from_supabase(target_food_code);
    // You can get data from "data" using JavaScript destructuring
    console.log(target_food_data);
    const food_name = target_food_data["food"];
    console.log(food_name);
    // document.getElementById("food_name").textContent = food_name;

    // Get nutrient values and update macronutrients
    const protein_amount = target_food_data["protein"];
    console.log(`Protein amount: ${protein_amount}`)
    document.getElementById("protein_amount").textContent = protein_amount + "g";

    const carbohydrate_amount = target_food_data["carbohydrate_by_difference"];
    console.log(`Carbohydrate amount: ${carbohydrate_amount}`)
    document.getElementById("carbohydrate_amount").textContent = carbohydrate_amount + "g";

    const fat_amount = target_food_data["total_lipid_fat"];
    console.log(`Fat amount: ${fat_amount}`)
    document.getElementById("fat_amount").textContent = fat_amount + "g";

    // End timer
    // var endTime = performance.now()

    // Log time
    // console.log(`Call to getFoodData took ${endTime - startTime} milliseconds`)
    // document.getElementById("time_taken").textContent = `${(endTime - startTime) / 1000} seconds`
}
