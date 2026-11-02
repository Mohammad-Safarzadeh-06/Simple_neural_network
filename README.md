# Simple_neural_network

## Project Overview

This project is a simple and practical implementation of an artificial neural network in pure Python designed to learn across different datasets. Built primarily for classification tasks, it gives you full flexibility to build your own custom neural network with any number of hidden layers and neurons, load your desired dataset, and feed it into the model for training. For error calculation, the model computes the squared difference between the network output and the target value, updating the parameters accordingly. You can also save the trained parameters to a file and reload them in future runs so you do not have to train the model from scratch every time.

## How to Use

* **Build a Custom Network:** Instantiate `neural_network_with_learning` with the input size, a list of hidden layer neuron counts, the output size, the activation function, and the learning rate:

  ```python
  model = neural_network_with_learning(2, [4, 4], 2, 'tanh', 0.01)
  ```

* **Select an Activation Function:** Choose between `'linear'`, `'tanh'`, or `'sigmoid'` for neuron activations and gradient calculations.

* **Train the Model:** Pass your input features and target outputs to the `train` method inside a loop to update weights and biases:

  ```python
  model.train(inputs, target_output)
  ```

* **Save and Load Parameters:** Extract the network parameters using `get_weight_list` and reload them into the model using `set_weights`:

  ```python
  weights = model.get_weight_list()
  model.set_weights(weights)
  ```

* **Run Predictions:** Pass new data to the `forward` method after training to compute the final output:

  ```python
  prediction = model.forward([x, y])
  ```

## Setup and Execution

1. **Prerequisites:** This script only uses Python's built-in `math` and `random` modules, requiring no external libraries.

2. **First-Time Setup:** The sample code reads `weights.txt` at startup, so comment out the `set_weights` line on your first run until the file is created.

3. **Run the Script:** Execute `nural_network.py` in your terminal to train the model on the sample data and test custom inputs interactively:

   ```bash
   python nural_network.py
   ```

## Project Status & Future Updates

This version of the project is still a work in progress. In future updates, saving and importing the model will be simplified to make storing and loading trained networks much easier.

---

# شبکه عصبی ساده (Simple_neural_network)

## معرفی پروژه

این پروژه یک پیاده‌سازی ساده و کاربردی از شبکه عصبی مصنوعی با پایتون خالص است که می‌تواند روی داده‌های مختلف آموزش ببیند و الگوها را یاد بگیرد. تمرکز اصلی این مدل روی مسائل دسته‌بندی (کلاسیفیکیشن) است و دست شما را برای طراحی ساختار شبکه کاملاً باز می‌گذارد؛ به طوری که می‌توانید شبکه عصبی دلخواه خودتان را با هر تعداد لایه پنهان و هر تعداد نورون بسازید، دیتاست مورد نظرتان را بازیابی کنید و برای آموزش به مدل بدهید. این مدل برای خطاگیری، اختلاف بین خروجی شبکه و جواب درست را به توان دو می‌رساند و بر اساس آن ضرایب را اصلاح می‌کند. همچنین می‌توانید ضرایب به دست آمده را در یک فایل ذخیره کنید و در دفعات بعدی مستقیماً داخل مدل بارگذاری کنید تا نیازی به آموزش دوباره از صفر نباشد.

## نحوه استفاده

* **ساخت شبکه دلخواه:** یک نمونه از کلاس `neural_network_with_learning` بسازید و تعداد ورودی‌ها، لیست نورون‌های لایه‌های پنهان، تعداد خروجی، تابع فعال‌سازی و نرخ یادگیری را مشخص کنید:

  ```python
  model = neural_network_with_learning(2, [4, 4], 2, 'tanh', 0.01)
  ```

* **انتخاب تابع فعال‌سازی:** بسته به نیازتان یکی از سه تابع `'linear'`، `'tanh'` یا `'sigmoid'` را برای محاسبات نورون‌ها و مشتق‌گیری انتخاب کنید.

* **آموزش مدل:** داده‌های ورودی و خروجی هدف را داخل یک حلقه به متد `train` بدهید تا وزن‌ها و بایاس‌ها با نرخ یادگیری تعیین‌شده اصلاح شوند:

  ```python
  model.train(inputs, target_output)
  ```

* **ذخیره و بارگذاری ضرایب:** با متد `get_weight_list` لیست ضرایب شبکه را بگیرید و در اجراهای بعدی با متد `set_weights` دوباره وارد مدل کنید:

  ```python
  weights = model.get_weight_list()
  model.set_weights(weights)
  ```

* **گرفتن خروجی:** بعد از اتمام آموزش، ورودی‌های جدید را به متد `forward` بدهید تا خروجی نهایی شبکه محاسبه شود:

  ```python
  prediction = model.forward([x, y])
  ```

## نصب و راه‌اندازی

1. **پیش‌نیازها:** این کد فقط از ماژول‌های پیش‌فرض پایتون (`math` و `random`) استفاده می‌کند و نیازی به نصب پکیج جانبی ندارد.

2. **تنظیم اجرای اول:** کد نمونه در ابتدا فایل `weights.txt` را می‌خواند؛ پس در اولین اجرا خط مربوط به `set_weights` را کامنت کنید تا اول این فایل ساخته شود.

3. **اجرای برنامه:** فایل `nural_network.py` را در ترمینال اجرا کنید تا شبکه روی داده‌های نمونه آموزش ببیند و در انتها بتوانید ورودی‌های دلخواه را تست کنید:

   ```bash
   python nural_network.py
   ```

## وضعیت پروژه و نسخه بعدی

این نسخه از پروژه هنوز کامل نیست و در حال توسعه قرار دارد. در آپدیت‌های بعدی، فرآیند ذخیره‌سازی و ایمپورت کردن مدل بازنویسی می‌شود تا ذخیره و فراخوانی شبکه بسیار راحت‌تر انجام شود.