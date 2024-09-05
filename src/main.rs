use std::fs;
use tract_onnx::prelude::*;


fn main() -> TractResult<()> {  

    let model = tract_onnx::onnx()
        .model_for_path("mobilenetv3_model.onnx")?
        .into_optimized()?
        .into_runnable()?;

    let mut true_positives: u32 = 0;
    let mut false_positives: u32 = 0;
    let mut true_negatives: u32 = 0;
    let mut false_negatives: u32 = 0;
    let mut total_images: u32 = 0;

    for entry in fs::read_dir("test/violence")? {
        let path = entry?.path();
        let image_path = path.to_str().unwrap();
        let image = image::open(image_path)?.to_rgb8();
        let resized = image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);    
        let image_tensor: Tensor = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
            resized[(x as _, y as _)][c] as f32 / 255.0
        }).into();   

        let result = model.run(tvec!(image_tensor.into()))?;
        let best = result[0]
            .to_array_view::<f32>()?
            .iter()
            .cloned()
            .zip(1..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let (_confidence, class_index) = best.unwrap();
        if class_index == 2{
            true_positives+=1;
        }else{
            false_negatives+=1;
        }
        total_images+=1
    }

    for entry in fs::read_dir("test/non_violence")? {
        let path = entry?.path();
        let image_path = path.to_str().unwrap();
        let image = image::open(image_path)?.to_rgb8();
        let resized = image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);    
        let image_tensor: Tensor = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
            resized[(x as _, y as _)][c] as f32 / 255.0
        }).into();   

        let result = model.run(tvec!(image_tensor.into()))?;
        let best = result[0]
            .to_array_view::<f32>()?
            .iter()
            .cloned()
            .zip(1..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let (_confidence, class_index) = best.unwrap();
        if class_index == 1{
            true_negatives+=1;
        }else{
            false_positives+=1;
        }
        total_images+=1
    }

    let correct_predictions = true_positives + true_negatives;
    let accuracy = correct_predictions as f32 / total_images as f32;    

    println!("True Positives (Violence): {true_positives}");
    println!("True Negatives (Non-Violence): {true_negatives}");
    println!("False Positives (Predicted Violence, Actual Non-Violence): {false_positives}");
    println!("False Negatives (Predicted Non-Violence, Actual Violence): {false_negatives}");
    println!("Total Images: {total_images}");
    println!("Accuracy: {:.2}%", accuracy * 100.0);
  
    Ok(())
}