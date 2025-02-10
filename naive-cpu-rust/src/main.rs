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

enum Axis {
    Row,
    Column,
}

#[derive(Clone)]
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
    fn zeros_like(temple_matrix: &Self) -> Self {
        Self::zeros(temple_matrix.rows_num, temple_matrix.cols_num)
    }
    fn ones(row: usize, col: usize) -> Self {
        Self {
            data: vec![1.0; row * col],
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

    fn add(&mut self, other: &Self) {
        assert_eq!(
            self.rows_num, other.rows_num,
            "Incompatible rows for addition: self.rows_num = {}, other.rows_num = {}",
            self.rows_num, other.rows_num
        );
        assert_eq!(
            self.cols_num, other.cols_num,
            "Incompatible columns for addition: self.cols_num = {}, other.cols_num = {}",
            self.cols_num, other.cols_num
        );
        let length = self.data.len();
        for i in 0..length {
            self.data[i] += other.data[i];
        }
    }

    fn sum(&self, axis: Axis) -> Self {
        match axis {
            //按行方向求和压缩
            Axis::Row => {
                let mut result_matrix = Self::zeros(1, self.cols_num);
                for col in 0..self.cols_num {
                    let mut tmp_sum = 0f32;
                    for row in 0..self.rows_num {
                        tmp_sum += self.get_item(row, col);
                    }
                    result_matrix.set_item(1, col, tmp_sum);
                }
                result_matrix
            }
            Axis::Column => {
                let mut result_matrix = Self::zeros(self.rows_num, 1);
                for row in 0..self.rows_num {
                    let mut tmp_sum = 0f32;
                    for col in 0..self.cols_num {
                        tmp_sum += self.get_item(row, col);
                    }
                    result_matrix.set_item(row, 1, tmp_sum);
                }
                result_matrix
            }
        }
    }
}

struct Linear {
    input_features: usize,
    output_features: usize,
    with_bias: bool, //留着，但是这里都按true来处理
    weights: Matrix,
    bias: Matrix,
    grad_weights: Matrix,
    grad_bias: Matrix,
    grad_input: Matrix,
}

impl Linear {
    fn new(input_features: usize, output_features: usize) -> Self {
        Self {
            input_features: input_features,
            output_features: output_features,
            with_bias: true,
            weights: Matrix::zeros(output_features, input_features),
            bias: Matrix::zeros(output_features, 1),
            grad_weights: Matrix::zeros(output_features, input_features),
            grad_bias: Matrix::zeros(output_features, 1),
            grad_input: Matrix::zeros(input_features, 1),
        }
    }
    fn initialize_weights(&mut self) {
        let mut rng = rand::rng();
        let weights_size = self.weights.data.len();
        let scale = f32::sqrt(2.0 / weights_size as f32);
        for i in 0..weights_size {
            self.weights.data[i] = rng.random::<f32>() * scale; //f32的random方法返回的就是0-1的数，应该是标准正态分布
        }
    }
    fn initialize_bias(&mut self) {
        //就是0，不用改
    }
    fn init_paramers(&mut self) {
        self.initialize_weights();
        self.initialize_bias();
    }

    fn forward(&self, input: &Matrix) -> Matrix {
        //z=wx+b
        let mut output = self.weights.multiply(input);
        output.add(&self.bias);
        output
    }

    fn backward(&mut self, grad_output: &Matrix, input: &Matrix) {
        //w[m,n]= y[m,batch_size]@x.T[batch_size,n]
        self.grad_weights = grad_output.multiply(&input.get_transpose_matrix());
        self.grad_bias = grad_output.sum(Axis::Column);
        // grad_x[n,1] = w.T[n,m] @ grad_out[m,1]
        self.grad_input = self.weights.get_transpose_matrix().multiply(grad_output);
    }
}

// trait SoftMax {
//     fn softmax_forward(&self) -> Self;
//     // fn softmax_backward(&self) -> Self;
// }

// trait ReLU {
//     fn relu_forward(&self) -> Self;
//     fn relu_backward(&self) -> Self;
// }

struct SoftMax {}
impl SoftMax {
    fn forward(&self, input: &Matrix) -> Matrix {
        //input[m,batch_size]
        //对其中的m个元素（一列）做softmax
        let batch_size = input.cols_num;
        let mut output = input.clone();
        for batch_index in 0..batch_size {
            //先找这一列最大的
            let mut max_element: f32 = input.get_item(0, batch_index);
            for row in 1..input.rows_num {
                if input.get_item(row, batch_index) > max_element {
                    max_element = input.get_item(row, batch_index);
                }
            }

            //算一下这一列减去最大值后的exp的和
            let mut sum: f32 = 0f32;
            for row in 0..input.rows_num {
                let exp_result = f32::exp(input.get_item(row, batch_index) - max_element);
                output.set_item(row, batch_index, exp_result);
                sum += exp_result;
            }

            //将结果除以这个sum
            for row in 0..input.rows_num {
                let finial_result = output.get_item(row, batch_index) / sum;
                output.set_item(row, batch_index, finial_result);
            }
        }
        output
    }
}

struct ReLU {}

impl ReLU {
    fn forward(&self, input: &Matrix) -> Matrix {
        let mut output = input.clone();
        for index in 0..output.data.len() {
            if output.data[index] < 0f32 {
                output.data[index] = 0f32;
            }
        }
        output
    }
    fn backward(&self, grad_output: &Matrix, input: &Matrix) -> Matrix {
        assert_eq!(
            grad_output.data.len(),
            input.data.len(),
            "Incompatitable grad_output and input"
        );
        let mut new_grad = grad_output.clone();
        for index in 0..new_grad.data.len() {
            if input.data[index] <= 0f32 {
                new_grad.data[index] = 0f32;
            }
        }
        new_grad
    }
}

struct CrossEntropyLoss {
    softmax: SoftMax,
}

impl CrossEntropyLoss {
    fn forward(&self, input: &Matrix, target: &Matrix) -> f32 {
        //target[batchsize,1]，每一个图像真实的label
        let input_probs = self.softmax.forward(input);
        //input_probs[10,batchsize]，每一个列向量是一个输出，表示对这个图像预测的各个数字概率大小
        assert_eq!(
            input.cols_num, target.rows_num,
            "The number of classes of input and target is not competitable"
        );

        // let mut correct_class_probs = Matrix::zeros_like(target);
        //correct_class_probs [batchsize,1]，每一个图像真实label预测正确的概率

        let mut log_sum = 0f32;
        for i in 0..target.rows_num {
            let max_prob_index = target.get_item(i, 0) as usize;
            let max_prob = input_probs.get_item(max_prob_index, i);
            // correct_class_probs.set_item(row, col, new_item);
            log_sum += f32::ln(max_prob);
        }
        log_sum / target.rows_num as f32
    }
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

    let one1 = Matrix::ones(4, 3);
    let mut two = Matrix::ones(4, 3);
    two.show();
    two.add(&one1);
    two.show();
}

fn test_linear() {
    let mut linear = Linear::new(3, 5);
    linear.init_paramers();
    let input = Matrix::ones(3, 1);
    input.show();
    let output = linear.forward(&input);
    output.show();
}

fn main() -> io::Result<()> {
    // test_dataset_load()?;
    // test_matrix_show();
    test_linear();
    Ok(())
}
