use std::fs;
use tract_onnx::prelude::*;
use image::{self, ImageBuffer, Rgb};
use ndarray::Array4;

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

    // Process violence images
    process_images("test/violence", true, &model, &mut true_positives, &mut false_positives, &mut true_negatives, &mut false_negatives, &mut total_images)?;

    // Process non-violence images
    process_images("test/non_violence", false, &model, &mut true_positives, &mut false_positives, &mut true_negatives, &mut false_negatives, &mut total_images)?;

    // Calculate accuracy
    let correct_predictions = true_positives + true_negatives;
    let accuracy = if total_images > 0 {
        correct_predictions as f32 / total_images as f32
    } else {
        0.0
    };

    println!("True Positives (Violence): {}", true_positives);
    println!("True Negatives (Non-Violence): {}", true_negatives);
    println!("False Positives (Predicted Violence, Actual Non-Violence): {}", false_positives);
    println!("False Negatives (Predicted Non-Violence, Actual Violence): {}", false_negatives);
    println!("Total Images: {}", total_images);
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}

fn process_images(dir: &str, is_violence: bool, model: &SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>, 
                  true_pos: &mut u32, false_pos: &mut u32, true_neg: &mut u32, false_neg: &mut u32, total: &mut u32) -> TractResult<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let image_path = path.to_str().unwrap();

        if image_path.ends_with(".png") || image_path.ends_with(".jpg") || image_path.ends_with(".jpeg") {
            let image = image::open(image_path)?.to_rgb8();
            let resized: ImageBuffer<Rgb<u8>, Vec<u8>> = image::imageops::resize(
                &image,
                224,
                224,
                image::imageops::FilterType::Lanczos3,
            );

            let image_array: Array4<f32> = Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
                let pixel = resized.get_pixel(x as u32, y as u32);
                pixel[c] as f32 / 255.0
            });

            // Convert Array4 to Tensor with explicit type annotation
            let image_tensor: Tensor = tract_ndarray::Array::from_shape_vec(
                (1, 224, 224, 3),
                image_array.into_raw_vec()
            )?.into();

            println!("Input Tensor Shape: {:?}", image_tensor.shape());

            let result = model.run(tvec!(image_tensor.into()))?;
            
            println!("Output shape: {:?}", result[0].shape());
            
            let best = result[0]
                .to_array_view::<f32>()?
                .iter()
                .cloned()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            if let Some((class_index, confidence)) = best {
                println!("Predicted class: {}, Confidence: {}", class_index, confidence);
                if (is_violence && class_index == 1) || (!is_violence && class_index == 0) {
                    if is_violence { *true_pos += 1; } else { *true_neg += 1; }
                } else {
                    if is_violence { *false_neg += 1; } else { *false_pos += 1; }
                }
                *total += 1;
            }
        }
    }
    Ok(())
}