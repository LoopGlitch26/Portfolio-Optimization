# Revolutionizing Portfolio Management in the age of Generative AI

### Introduction

Traditional portfolio management has relied on human expertise and heuristics to select assets and optimize portfolios.
The combination of **deep reinforcement learning (DRL)** and **generative adversarial networks (GANs)** provides a robust framework for portfolio management.
The aim of the project is to revolutionize portfolio management in the age of Generative AI and demonstrate the potential of DRL and GANs to improve portfolio performance and manage risk. 
This approach enables the creation of synthetic data that can be utilized to train an agent, resulting in improved investment  decision-making.

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/60ebd501-a1c7-4ad8-8b14-a8925e3fe397)

### Problem Identification

Traditional portfolio management relies on human expertise and heuristics, which can be suboptimal and lead to losses.
Market conditions are constantly changing, making it difficult to adapt and optimize portfolio strategies in a timely manner.

### Solution Approach

Develop a GAN-based synthetic dataset by forecasting market conditions to simulate multiple investment scenarios and reduce the impact of real-time market fluctuations. 
Train a DRL model on the synthetic dataset to identify low-risk high-return investment opportunities and optimize portfolio performance by continuously adapting to evolving market conditions.

### Model Diagram

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/243b968e-33de-4036-8bfe-ff2d2585c0f8)

### Dataset Description

The dataset contains the stock data of FAANG companies from 1st January 2015. It was sourced from a dataset available on [Kaggle](https://www.kaggle.com/datasets/kaushiksuresh147/faang-fbamazonapplenetflixgoogle-stocks?resource=download) and [Yahoo Finance](https://finance.yahoo.com/?guccounter=1) that contains daily stock market data for **Facebook, Amazon, Apple, Netflix, and Google**.

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/8aed0b75-6071-4a90-863c-9053c7f41f6e)

Parameters used:
* Basic Variables: Highest Price, Lowest Price, Opening Price, Closing Price, Volume
* Technical Indicators: Simple Moving Average (SMA), Exponential Moving Average (EMA), Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), Average True Range (ATR), Bollinger Bands, Raw Stochastic Value (RSV) 

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/8c7d64eb-0054-47b2-8441-95f828908ada)

### Flowchart

Data Preprocessing:

![Screenshot 2023-06-17 at 5 21 44 AM](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/6d6981a3-d71a-445a-ae7b-b80ed31164d2)

GAN + DRL:

![Screenshot 2023-06-15 at 4 36 22 PM](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/17794f12-deca-49b0-a956-81f57e4defee)

### Portfolio Prediction

Facebook: 

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/61d768c1-858b-4046-9100-6f7fd7566029)
![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/c59716a2-df24-48f2-8d70-ce59d2640246)

Amazon:

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/aa4b3a69-76fd-4a29-9b07-3d59b61d58ed)
![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/ca55ebc3-09bc-402b-b217-db7078454ccc)

Apple: 

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/e86aa9df-6cd5-4e4d-aa9a-6160c5457ea3)
![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/eb434d93-0061-4807-9a08-e67506e852a5)

Netflix:

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/b65dbf75-7409-452e-8944-63739f641fed)
![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/1b951907-b3dd-4305-9b16-c34fc00529af)

Google: 

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/eb959868-fae0-4d1d-aa06-2468036c9fb8)
![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/7fe94136-a8d9-4e35-bf58-c9c9e302dcdd)

### Result Analysis

GAN Loss Plot:

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/e430cc78-20d1-469a-95f1-33b7f36da209)

Portfolio Rewards:

![image](https://github.com/LoopGlitch26/Portfolio-Optimization-using-Generative-AI/assets/53336715/9dcf1baa-4066-457d-b68a-e99f81c96320)

### Future Scope

* Analyze news articles, social media feeds, and other textual data to extract sentiment and assess market sentiment towards FAANG companies, leveraging the use of NLP techniques.

* Conduct additional comparative analysis of the DRL and GAN's combined model with other advanced existing systems, exploring its performance across different market conditions and asset classes.

* Enhance the synthetic asset return generation capabilities of the proposed system by incorporating more sophisticated algorithms and techniques to provide even better insights for investment decisions.



