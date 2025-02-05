use rand::Rng;
use std::fs;
use std::io::{self, Read};

const PICTURE_WIDTH: usize = 28;
const PICTURE_HEIGHT: usize = 28;
const PICTURE_SIZE: usize = PICTURE_WIDTH * PICTURE_HEIGHT;
const LABEL_SIZE: usize = 1;
const TRAIN_SIZE: usize = 6e4 as usize;
const TEST_SIZE: usize = 1e4 as usize;

struct DataSet {
    images_file_path: String,
    labels_file_path: String,
    num_elements: usize,
    images_data: Option<Vec<f32>>,
    labels_data: Option<Vec<i32>>,
}
impl DataSet {
    fn load_images_data(&mut self) -> io::Result<()> {
        let mut data_bin_file = fs::File::open(&self.images_file_path)?;
        let mut buffer = vec![0u8; self.num_elements * std::mem::size_of::<f32>() * PICTURE_SIZE];
        data_bin_file.read_exact(&mut buffer)?;
        let data: Vec<f32> = buffer
            .chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        self.images_data = Some(data);
        Ok(())
    }

    fn load_labels_data(&mut self) -> io::Result<()> {
        let mut data_bin_file = fs::File::open(&self.labels_file_path)?;
        let mut buffer = vec![0u8; self.num_elements * std::mem::size_of::<i32>() * LABEL_SIZE];
        data_bin_file.read_exact(&mut buffer)?;
        let data: Vec<i32> = buffer
            .chunks_exact(std::mem::size_of::<i32>())
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        self.labels_data = Some(data);
        Ok(())
    }
    fn load_data(&mut self) -> io::Result<()> {
        self.load_images_data()?;
        self.load_labels_data()?;
        if let (Some(images_data), Some(labels_data)) = (&self.images_data, &self.labels_data) {
            println!(
                "Data loaded, images length: {}, labels length: {}.",
                { images_data.len() },
                { labels_data.len() }
            );
        } else {
            println!("Data load failed!");
        }
        Ok(())
    }

    fn show(&self, index: usize) {
        println!("The index is: {}", { index });
        if let Some(labels_data) = &self.labels_data {
            println!("The picture of {} is as follows:", { labels_data[index] });
        } else {
            println!("There is no picture");
        }
        if let Some(images_data) = &self.images_data {
            for i in 0..PICTURE_HEIGHT {
                for j in 0..PICTURE_WIDTH {
                    if images_data[index * PICTURE_SIZE + i * PICTURE_WIDTH + j] > 0f32 {
                        print!("x");
                    } else {
                        print!(" ");
                    }
                }
                print!("\n");
            }
        } else {
            println!("There is no picture");
        }
    }
    fn show_random(&self) {
        let mut rng = rand::rng();
        let index = rng.random_range(0..self.num_elements);
        self.show(index);
    }
}

fn main() -> io::Result<()> {
    let mut train_dataset: DataSet = DataSet {
        images_file_path: "../mnist_data/X_train.bin".to_string(),
        labels_file_path: "../mnist_data/y_train.bin".to_string(),
        num_elements: TRAIN_SIZE,
        images_data: None,
        labels_data: None,
    };
    train_dataset.load_data()?;
    train_dataset.show_random();
    let mut test_dataset: DataSet = DataSet {
        images_file_path: "../mnist_data/X_test.bin".to_string(),
        labels_file_path: "../mnist_data/y_test.bin".to_string(),
        num_elements: TEST_SIZE,
        images_data: None,
        labels_data: None,
    };
    test_dataset.load_data()?;
    test_dataset.show_random();
    Ok(())
}
