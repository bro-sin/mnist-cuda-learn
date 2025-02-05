use std::fs;
use std::io::{self, Read};

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
        let mut buffer = vec![0u8; self.num_elements * 4 * 28 * 28];
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
        let mut buffer = vec![0u8; self.num_elements * std::mem::size_of::<i32>()];
        data_bin_file.read_exact(&mut buffer)?;
        let data: Vec<i32> = buffer
            .chunks_exact(std::mem::size_of::<i32>())
            .map(|chunk| {
                let mut tmp = [0u8; 4];
                tmp.copy_from_slice(chunk);
                i32::from_le_bytes(tmp)
            })
            .collect();
        self.labels_data = Some(data);
        Ok(())
    }
    fn load_data(&mut self) -> io::Result<()> {
        self.load_images_data();
        self.load_labels_data();
        if let (Some(images_data), Some(labels_data)) = (&self.images_data, &self.labels_data) {
            print!(
                "Data loaded, images length: {}, labels length: {}.\n",
                { images_data.len() },
                { labels_data.len() }
            );
        } else {
            print!("Data load failed!");
        }
        Ok(())
    }

    fn show(&self, index: usize) {
        if let Some(labels_data) = &self.labels_data {
            // for i in 0..labels_data.len() {
            //     println!("labels[{}]:{}", { i }, { labels_data[i] });
            // }
            print!("The picture of {} is as follows:\n", { labels_data[index] });
        }
        //在终端显示第一张图片
        if let Some(images_data) = &self.images_data {
            for i in 0..28 {
                for j in 0..28 {
                    if images_data[index * 28 * 28 + i * 28 + j] > 0f32 {
                        print!("x");
                    } else {
                        print!(" ");
                    }
                }
                print!("\n");
            }
        }
    }
}

fn main() -> io::Result<()> {
    let mut train_dataset: DataSet = DataSet {
        images_file_path: "../mnist_data/X_train.bin".to_string(),
        labels_file_path: "../mnist_data/y_train.bin".to_string(),
        num_elements: 10000,
        images_data: None,
        labels_data: None,
    };
    train_dataset.load_data();
    train_dataset.show(89);
    Ok(())
}
