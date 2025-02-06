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

struct Matrix {
    data: Vec<f32>,
    row_major: bool,
    rows_num: usize,
    cols_num: usize,
}

impl Matrix {
    fn zeros(row: usize, col: usize) -> Self {
        Self {
            data: vec![0.0; row * col],
            row_major: true,
            rows_num: row,
            cols_num: col,
        }
    }
    fn transpose(&mut self) {
        self.row_major = !self.row_major;
        (self.rows_num, self.cols_num) = (self.cols_num, self.rows_num);
    }
    fn get_transpose_matrix(&self) -> Self {
        Self {
            data: self.data.clone(),
            row_major: !self.row_major,
            rows_num: self.cols_num,
            cols_num: self.rows_num,
        }
    }
    fn get_index(&self, row: usize, col: usize) -> usize {
        if self.row_major {
            return row * self.cols_num + col;
        } else {
            return col * self.rows_num + row;
        }
    }
    fn get_item(&self, row: usize, col: usize) -> f32 {
        return self.data[self.get_index(row, col)];
    }
    fn set_item(&mut self, row: usize, col: usize, new_item: f32) {
        let _index = self.get_index(row, col);
        self.data[_index] = new_item;
    }
    fn show(&self) {
        println!(
            "The matrix with {} rows and {} columns is as follows:",
            self.rows_num, self.cols_num
        );
        for row in 0..self.rows_num {
            for col in 0..self.cols_num {
                print!("{:5.2}\t", self.get_item(row, col));
            }
            print!("\n");
        }
        print!("\n");
    }
    fn multiply(&self, other: &Self) -> Self {
        assert_eq!(
            self.cols_num, other.rows_num,
            "Incompatible matrix dimensions for multiplication"
        );
        let mut result_matrix = Self::zeros(self.rows_num, other.cols_num);
        for row in 0..self.rows_num {
            for col in 0..other.cols_num {
                let result_index = result_matrix.get_index(row, col);
                for dot_index in 0..self.cols_num {
                    result_matrix.data[result_index] +=
                        self.get_item(row, dot_index) * other.get_item(dot_index, col);
                }
            }
        }
        result_matrix
    }
}

struct Linear {
    input_features: usize,
    output_features: usize,
    with_bias: bool,
    weights: Vec<f32>,
    bias: Vec<f32>,
    grad_weights: Vec<f32>,
    grad_bias: Vec<f32>,
    grad_input: Vec<f32>,
}

impl Linear {
    fn initialize_weights(&mut self) {
        let mut rng = rand::rng();
        // let mut weights = &self.weights;
        let weights_size = self.weights.len();
        let scale = f32::sqrt(2.0 / weights_size as f32);
        for i in 0..weights_size {
            // self.weights[i] = rng.random_range(0.0..1.0) * scale; //这里能不能调用默认的random?确认返回的是不是0-1之间的数
            self.weights[i] = rng.random::<f32>() * scale; //f32的random方法返回的就是0-1的数，应该是标准正态分布
        }
    }
    fn initialize_bias(&mut self) {
        for i in 0..self.bias.len() {
            self.bias[i] = 0f32;
        }
    }
    fn forward(&self, input: &Vec<f32>, output: &mut Vec<f32>) {}
}

fn test_dataset_load() -> io::Result<()> {
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

fn test_matrix_show() {
    let mut matrix = Matrix {
        data: vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        row_major: true,
        rows_num: 3,
        cols_num: 4,
    };
    matrix.show();
    matrix.transpose();
    matrix.show();

    let mut test_zeros = Matrix::zeros(5, 6);
    test_zeros.show();
    test_zeros.transpose();
    test_zeros.show();

    let b = matrix.get_transpose_matrix();
    let c = b.multiply(&matrix);
    c.show();
}

fn main() -> io::Result<()> {
    test_dataset_load()?;
    test_matrix_show();
    Ok(())
}
