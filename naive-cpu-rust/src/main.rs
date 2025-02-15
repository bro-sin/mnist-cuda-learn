use rand::Rng;
use std::fs;
use std::io::{self, Read};

const PICTURE_WIDTH: usize = 28;
const PICTURE_HEIGHT: usize = 28;
const PICTURE_SIZE: usize = PICTURE_WIDTH * PICTURE_HEIGHT;
const HIDDEN_SIZE: usize = 256;
const LABEL_SIZE: usize = 1;
const TRAIN_SIZE: usize = 1e4 as usize;
const TEST_SIZE: usize = 1e3 as usize;
const BATCH_SIZE: usize = 4; //每一次训练使用的样本数
const LEARNING_RATE: f32 = 1e-3;

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

    fn get_train_matrix(&self, index: usize) -> (Option<Matrix>, Option<Matrix>) {
        //需要定义BATCH_SIZE:一个batch（一次更新参数）有多少张图片和label（多少个样本）用来训练
        //这里index应当是batch_index,也就是在num_batches这么多个batch中的第几个
        //这个函数就是取出第index个batch的图片和label，总共有BATCH_SIZE个样本

        if let (Some(images_data), Some(labels_data)) = (&self.images_data, &self.labels_data) {
            let train_labels = Matrix {
                data: labels_data[BATCH_SIZE * index..BATCH_SIZE * (index + 1)]
                    .iter()
                    .map(|&label| label as f32)
                    .collect(),
                row_major: true,
                rows_num: BATCH_SIZE,
                cols_num: 1,
            };
            let train_images = Matrix {
                data: images_data
                    [BATCH_SIZE * PICTURE_SIZE * index..BATCH_SIZE * PICTURE_SIZE * (index + 1)]
                    .to_vec(),
                row_major: true,
                rows_num: PICTURE_SIZE,
                cols_num: BATCH_SIZE,
            };
            (Some(train_images), Some(train_labels))
        } else {
            (None, None)
        }
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

    fn multiply_scale(&mut self, other: f32) {
        for i in 0..self.data.len() {
            self.data[i] *= other;
        }
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
        // let length = self.data.len();
        // for i in 0..length {
        //     self.data[i] += other.data[i];
        // }
        for row in 0..self.rows_num {
            for col in 0..self.cols_num {
                let new_item = self.get_item(row, col) + other.get_item(row, col);
                self.set_item(row, col, new_item);
            }
        }
    }
    fn add_bias(&mut self, other: &Self) {
        //考虑到bias和前面输出的维度并不一致，前面输出会有一个BATCH_SIZE，而bias是[output_features,1]
        //所以这里的add_bias是将bias的每一列加到self的每一列上
        //要求行是一样的，列不作要求
        assert_eq!(
            self.rows_num, other.rows_num,
            "Incompatible rows for addition: self.rows_num = {}, other.rows_num = {}",
            self.rows_num, other.rows_num
        );
        let cols_num = self.cols_num;
        for row in 0..other.rows_num {
            let bias_value = other.get_item(row, 0);
            for col in 0..cols_num {
                let new_item = self.get_item(row, col) + bias_value;
                self.set_item(row, col, new_item);
            }
        }
    }

    fn subtract(&mut self, other: &Self) {
        assert_eq!(
            self.rows_num, other.rows_num,
            "Incompatible rows for subtraction: self.rows_num = {}, other.rows_num = {}",
            self.rows_num, other.rows_num
        );
        assert_eq!(
            self.cols_num, other.cols_num,
            "Incompatible columns for subtraction: self.cols_num = {}, other.cols_num = {}",
            self.cols_num, other.cols_num
        );
        // let length = self.data.len();
        // for i in 0..length {
        //     self.data[i] -= other.data[i];
        // }
        for row in 0..self.rows_num {
            for col in 0..self.cols_num {
                let new_item = self.get_item(row, col) - other.get_item(row, col);
                self.set_item(row, col, new_item);
            }
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
                    result_matrix.set_item(0, col, tmp_sum);
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
                    result_matrix.set_item(row, 0, tmp_sum); //记住索引是0
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
        //output的维度是[output_features,BATCH_SIZE]
        //bias的维度是[output_features,1]
        output.add_bias(&self.bias);
        output
    }

    fn backward(&mut self, grad_output: &Matrix, input: &Matrix) {
        //w[m,n]= y[m,BATCH_SIZE]@x.T[BATCH_SIZE,n]
        self.grad_weights = grad_output.multiply(&input.get_transpose_matrix());
        self.grad_bias = grad_output.sum(Axis::Column);
        // grad_x[n,1] = w.T[n,m] @ grad_out[m,1]
        self.grad_input = self.weights.get_transpose_matrix().multiply(grad_output);
    }

    fn update_weights(&mut self, learning_rate: f32) {
        //w=w-grad_w*learning_rate
        self.grad_weights.multiply_scale(learning_rate);
        self.weights.subtract(&self.grad_weights);
        //b=b-grad_b*learning_rate
        self.grad_bias.multiply_scale(learning_rate);
        self.bias.subtract(&self.grad_bias);
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
        //input[m,BATCH_SIZE]
        //对其中的m个元素（一列）做softmax
        let batch_size = input.cols_num;
        let mut output = input.clone();
        for batch_index in 0..batch_size {
            // 这里batch_index指的是在batch_size这么多个样本中的第几个
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
    fn new() -> Self {
        Self {
            softmax: SoftMax {},
        }
    }
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
        -log_sum / target.rows_num as f32 // 交叉熵是算的对数平均的相反数
    }
}

struct MLP {
    input_features: usize,
    hidden_features: usize,
    num_classes: usize,
    fc1: Linear,
    relu: ReLU,
    fc2: Linear,
    softmax: SoftMax,
    cross_entropy_loss: CrossEntropyLoss,
}

impl MLP {
    fn new(input_features: usize, hidden_features: usize, num_classes: usize) -> Self {
        let mut this_mlp = Self {
            input_features,
            hidden_features,
            num_classes,
            fc1: Linear::new(input_features, hidden_features),
            relu: ReLU {},
            fc2: Linear::new(hidden_features, num_classes),
            softmax: SoftMax {},
            cross_entropy_loss: CrossEntropyLoss::new(),
        };
        this_mlp.fc1.init_paramers();
        this_mlp.fc2.init_paramers();
        this_mlp
    }

    fn forward(&self, input: &Matrix) -> Vec<Matrix> {
        //input: 28*28,BATCH_SIZE
        let fc1_output = self.fc1.forward(input);
        let relu_output = self.relu.forward(&fc1_output);
        let fc2_output = self.fc2.forward(&relu_output);
        vec![fc2_output, input.clone(), fc1_output, relu_output]
    }

    fn backward(&mut self, grad_output: &Matrix, cache: Vec<Matrix>) {
        let (fc1_input, fc1_output, relu_output) = (&cache[1], &cache[2], &cache[3]);

        self.fc2.backward(grad_output, relu_output);
        let grad_fc2 = &self.fc2.grad_input;

        let grad_relu = self.relu.backward(grad_fc2, fc1_output);

        self.fc1.backward(&grad_relu, fc1_input);
    }

    fn update_weights(&mut self, learning_rate: f32) {
        self.fc1.update_weights(learning_rate);
        self.fc2.update_weights(learning_rate);
    }
}

impl MLP {
    fn train(&mut self, train_data: &DataSet, learning_rate: f32, epochs: usize) {
        let num_batches = train_data.num_elements / BATCH_SIZE;
        for epoch in 0..epochs {
            //训练的轮数
            let mut total_loss = 0f32;
            let mut correct: u32 = 0;
            for batch_index in 0..num_batches {
                //这里batch_index指的是在num_batches这么多个batch中的第几个
                //将数据分成多个batch，每一次由BATCH_SIZE个样本进行前向传播和反向传播，然后更新参数
                if let (Some(train_images), Some(train_labels)) =
                    train_data.get_train_matrix(batch_index)
                {
                    let cache = self.forward(&train_images);
                    //根据crossentropyloss的反向传播，计算出grad_output
                    let y_pred = &cache[0]; //这是没有经过softmax的输出
                    let loss = self.cross_entropy_loss.forward(y_pred, &train_labels);
                    total_loss += loss;
                    let mut softmax_probs = self.softmax.forward(y_pred);

                    //计算正确预测的个数
                    for i in 0..BATCH_SIZE {
                        let label = train_labels.get_item(i, 0) as usize;
                        let mut predicted_index = 0;
                        for j in 1..10 {
                            if softmax_probs.get_item(j, i)
                                > softmax_probs.get_item(predicted_index, i)
                            {
                                predicted_index = j;
                            }
                        }
                        if predicted_index == label {
                            correct += 1;
                        }
                    }

                    let mut y_true_one_hot = Matrix::zeros_like(&softmax_probs);
                    for i in 0..BATCH_SIZE {
                        let label = train_labels.get_item(i, 0) as usize;
                        //label就是第i个样本的真实label
                        y_true_one_hot.set_item(label, i, 1f32);
                        //这里将y_true_one_hot的第i个样本的概率设置为1,也就是我们softmax_probs理想的输出
                    }
                    softmax_probs.subtract(&y_true_one_hot);
                    //softmax_probs-y_true_one_hot就是crossentropy对fc2的输出的梯度
                    let grad_output = softmax_probs;
                    self.backward(&grad_output, cache);
                    let mut _tmp_grad_bias = self.fc1.grad_bias.get_item(200, 0);
                    let _tmp_grad_weight = self.fc1.grad_weights.get_item(200, 700);
                    println!(
                        "grad_tmp_bias:{}, grad_tmp_weight:{}",
                        _tmp_grad_bias, _tmp_grad_weight
                    );

                    self.update_weights(learning_rate);
                    let _tmp_bias = self.fc1.bias.get_item(200, 0);
                    let _tmp_weight = self.fc1.weights.get_item(200, 700);
                    println!("tmp_bias:{}, tmp_weight:{}", _tmp_bias, _tmp_weight);

                    //输出当前训练进度
                    // if batch_index % 100 == 0 {
                    println!(
                        "Epoch: {}/{}, Iter: {}/{}, Loss: {:.4}, Accuracy: {:.4}%",
                        epoch,
                        epochs,
                        batch_index,
                        num_batches,
                        total_loss / (batch_index + 1) as f32,
                        (100 * correct) as f32 / ((batch_index + 1) * BATCH_SIZE) as f32
                    );
                    // }
                } else {
                    println!("There is no  available data");
                }
            }
        }
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

fn test_MLP() -> io::Result<()> {
    let mut mlp = MLP::new(PICTURE_SIZE, HIDDEN_SIZE, 10);
    let mut train_dataset: DataSet = DataSet {
        images_file_path: "../mnist_data/X_train.bin".to_string(),
        labels_file_path: "../mnist_data/y_train.bin".to_string(),
        num_elements: TRAIN_SIZE,
        images_data: None,
        labels_data: None,
    };
    train_dataset.load_data()?;
    mlp.train(&train_dataset, LEARNING_RATE, 5);

    Ok(())
}

fn main() -> io::Result<()> {
    // test_dataset_load()?;
    // test_matrix_show();
    // test_linear();
    test_MLP();
    Ok(())
}
