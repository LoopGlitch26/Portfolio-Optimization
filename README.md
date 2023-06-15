# Senior Design Project (B.Tech in CSE)

## Revolutionizing Portfolio Management in the age of Generative AI
### Introduction

Traditional portfolio management has relied on human expertise and heuristics to select assets and optimize portfolios.
The combination of **deep reinforcement learning (DRL)** and **generative adversarial networks (GANs)** provides a robust framework for portfolio management.
The aim of the project is to revolutionize portfolio management in the age of Generative AI and demonstrate the potential of DRL and GANs to improve portfolio performance and manage risk. 
This approach enables the creation of synthetic data that can be utilized to train an agent, resulting in improved investment  decision-making.

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/783569d9-0d14-4bd1-b42f-789f98719fcb)

### Problem Identification

Traditional portfolio management relies on human expertise and heuristics, which can be suboptimal and lead to losses.
Market conditions are constantly changing, making it difficult to adapt and optimize portfolio strategies in a timely manner.

### Solution Approach

To develop a GAN-based synthetic dataset by forecasting market conditions to simulate multiple investment scenarios and reduce the impact of real-time market fluctuations. 
To train a DRL model on the synthetic dataset to identify low-risk high-return investment opportunities and optimize portfolio performance by continuously adapting to evolving market conditions.

### Model Diagram

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/5930e6e6-0538-49e3-87a8-65482d662994)

### Dataset Description

The dataset contains the stock data of FAANG companies from 1st January 2015. It was sourced from a dataset available on [Kaggle](https://www.kaggle.com/datasets/kaushiksuresh147/faang-fbamazonapplenetflixgoogle-stocks?resource=download) and [Yahoo Finance](https://finance.yahoo.com/?guccounter=1) that contains daily stock market data for **Facebook, Amazon, Apple, Netflix, and Google**.

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/c42a6674-7f17-45d5-8174-0a3fc49f9362)

Parameters used:
* Basic Variables: Highest Price, Lowest Price, Opening Price, Closing Price, Volume
* Technical Indicators: Simple Moving Average (SMA), Exponential Moving Average (EMA), Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), Average True Range (ATR), Bollinger Bands, Raw Stochastic Value (RSV) 

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/17092d23-31a1-4b1f-ab84-2173c6158a83)

### Flowchart

Data Preprocessing:

![Screenshot 2023-06-15 at 4 30 50 PM](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/0d894966-e62b-455c-b417-abf6d9603569)

GAN + DRL:

![Screenshot 2023-06-15 at 4 36 22 PM](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/a42f8de3-9275-4ddd-8e0f-e20cde802f58)

### Portfolio Prediction

Facebook: 

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/195f4b82-7f0d-4edc-9659-c0883f2898e6)
![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/0b24a9be-31bb-4b1b-8826-819804d78067)

Amazon:

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/775a4fc0-273d-42b1-9054-695d63bacb97)
![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/fe60c64f-55ad-43b5-af4f-a5d98f3d01e2)

Apple: 

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/14190206-794d-444d-bdc1-9fe186a8d616)
![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/aca235b3-30ff-4392-991a-80f6b845ed5f)

Netflix:

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/843f70f4-75ae-4789-944f-e89e480245c3)
![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/3b9539af-b326-4d09-8beb-8765a595e9bb)

Google: 

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/76a1137b-80a4-4f00-b514-f48b5e33ef84)
![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/8b0e8236-8022-467a-9cdb-4686feead10c)

### Result Analysis

GAN Loss Plot:

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/057b0de1-ba74-4dc6-889b-cef571bb51d9)

Portfolio Rewards:

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/9de5e76b-1ba4-409a-8bbc-a87a43846384)

### Future Scope

* Analyze news articles, social media feeds, and other textual data to extract sentiment and assess market sentiment towards FAANG companies, leveraging the use of NLP techniques.

* Conduct additional comparative analysis of the DRL and GAN's combined model with other advanced existing systems, exploring its performance across different market conditions and asset classes.

* Enhance the synthetic asset return generation capabilities of the proposed system by incorporating more sophisticated algorithms and techniques to provide even better insights for investment decisions.

### Team Members

* Bravish Ghosh (Team Lead)
* Tariq Nasar
* Pratik Gupta
* Aayush Kumar
* Dr. Binayak Panda (Supervisor)





