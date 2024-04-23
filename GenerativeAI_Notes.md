# 前言
快速筆記一下上課內容，以利日後Ctrl+F搜尋keywords和concept

# 2024.02.24 【生成式AI導論 2024】第1講：生成式AI是什麼？ 
- video: https://www.youtube.com/watch?v=JGtqpQXfJis
- slide: https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring-course-data/0223/0223_intro_gai.pdf
- 生成式人工智慧 ⊂ 人工智慧
  - 人工智慧 (Artificial Intelligence, AI): 讓機器展現「智慧」
  - 生成式人工智慧 (Generative AI, GenAI): 機器產生複雜有結構的物件
    - 文章-由文字所構成, 影像-由像素所組成, 語音-由取樣點構成
    - i.g. ChatGPT 用 Transformer
    - i.g. Stable Diffusion, Midjourney, DALL·E 用 Diffusion Model
- 生成式人工智慧 ⊂ 深度學習 ⊂ 機器學習 ⊂ 人工智慧
  - 機器學習 (Machine Learning) ≈ 機器自動從資料找一個函式
    - i.g. 函式 𝑦 = 𝑓(𝑥) = 𝑎𝑥 + 𝑏, a和b 為 參數 (Parameter)
    - i.g. 機器學習可以把有上萬個參數的函式的參數找出來 透過 訓練, training (學習, learning)
    - i.g. 有了函式後計算答案叫 測試, testing (推論, inference)
  - 深度學習(Deep Learning) 是一種機器學習技術, 使用 類神經網路 (Neural Network)
  - <img src="https://i.imgur.com/QTlrJ3k.png" height=200/>
- 生成策略: 文字接龍 
  - 機器需要能夠產生在訓練時從來沒有看過的東西
    - 原本生文章 可能性 窮盡無盡!
    - 拆解成一連串 文字接龍 or 像素接龍 分類問題 答案有限!
  - 複雜的物件 拆解成較小的單位 依照某種固定的順序依序生成 叫Autoregressive Generation

# 2024.03.03 【生成式AI導論 2024】第2講：今日的生成式人工智慧厲害在哪裡？從「工具」變為「工具人」 
- video: https://www.youtube.com/watch?v=glBhOQ1_RkE
- slide: https://drive.google.com/file/d/1Ru6DUX8KrSzCvn2DN1-YluTyx5rw3QD3/view
- 今日的生成式人工智慧厲害在哪裡? 功能單一 -> 沒有特定功能(通用型)
  - GPT系列 為OpenAI所開發
  - Gemini 為Google所開發
  - Llama系列 為Meta釋出的開源大型語言模型
  - TAIDE模型為Llama2模型結合臺灣文化與正體中文語料之衍生模型 (來自國科會-推動可信任生成式AI發展先期計畫)
- 可能的研究方向:
  - 評估模型(evaluation)困難 不好評估答案之於問題是否完美被解決
  - 要防止說出有害內容
  - Improvement
    - 改變自己來強化模型 (improving inputs) - prompt engineering, [continue](https://github.com/shannon112/MareepLearning/blob/master/GenerativeAI_Notes.md#20240303%E7%94%9F%E6%88%90%E5%BC%8Fai%E5%B0%8E%E8%AB%96-2024%E7%AC%AC3%E8%AC%9B%E8%A8%93%E7%B7%B4%E4%B8%8D%E4%BA%86%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E4%BD%A0%E5%8F%AF%E4%BB%A5%E8%A8%93%E7%B7%B4%E4%BD%A0%E8%87%AA%E5%B7%B1-%E4%B8%8A--%E7%A5%9E%E5%A5%87%E5%92%92%E8%AA%9E%E8%88%87%E6%8F%90%E4%BE%9B%E6%9B%B4%E5%A4%9A%E8%B3%87%E8%A8%8A)
    - 訓練自己的模型 (improving parameters)

# 2024.03.03【生成式AI導論 2024】第3講：訓練不了人工智慧？你可以訓練你自己 (上) — 神奇咒語與提供更多資訊 
- video: https://www.youtube.com/watch?v=A3Yx35KrSN0&t=1722s
- slide: https://drive.google.com/file/d/1JTexyex5hrHmNdrkXy-jOVKZlycODC7Y/view
- 改變自己來強化模型 (improving inputs) - prompt engineering
  - <img src="https://i.imgur.com/EekRhP4.png" height=200>
  - 1.神奇咒語 (不一定對所有模型、所有任務都適用) => 最有名: Chain of Thought 
    - 叫模型思考 "Chain of Thought (CoT)"
      - i.g. Let's think step by step, 
      - Large Language Models are Zero-Shot Reasoners, https://arxiv.org/abs/2205.11916
    - 叫模型解釋一下自己的答案
      - i.g. Answer by starting with Analysis 
      - A Closer Look into Automatic Evaluation Using Large Language Models, https://arxiv.org/abs/2310.05657
      - Can Large Language Models Be an Alternative to Human Evaluations?, https://arxiv.org/abs/2305.01937
      - The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning, https://arxiv.org/abs/2205.03401
    - 情緒勒索
      - i.g. This is very important to my career 
      - Large Language Models Understand and Can be Enhanced by Emotional Stimuli, https://arxiv.org/abs/2307.11760
    - 更多的神奇咒語 驗證都市傳說
      - i.g. No need to be polite like “please”, “if you don’t mind”, “thank you”, “I would like to”, etc., 有禮貌是沒用的
      - i.g. Employ affirmative directives such as ‘do,’ while steering clear of negative language like ‘don’t’. 正面表述 好過負面表述
      - i.g. Add “I’m going to tip $xxx for a better solution!” 說要給小費 是有用的
      - i.g. Incorporate the following phrases: “You will be penalized” 說會有處罰 是有用的
      - i.g. Add prompt “Ensure that your answer is unbiased and avoids relying on stereotypes.” 要其中立無偏見 是有用的
      - Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4, https://arxiv.org/abs/2312.16171
    - 用增強式學習 (Reinforcement Learning, RL) 找神奇咒語
      - i.g. 任務目標:回應越長越好, prompt: “ways ways ways ways ways ways ways .......”
      - Learning to Generate Prompts for Dialogue Generation through Reinforcement Learning, https://arxiv.org/abs/2206.03931
    - 用大型語言模型來 找神奇咒語
      - i.g. Let’s work this out in a step by step way to be sure we have the right answer. 
      - i.g. Take a deep breath and work on this problem step-by-step  
      - Large Language Models Are Human-Level Prompt Engineers, https://arxiv.org/abs/2211.01910
  - 2.把前提講清楚 => 最有名: In-context Learning
    - 提供生成式AI原本不清楚的資訊
    - 提供範例 In-context Learning - Language Models are Few-Shot Learners, https://arxiv.org/abs/2005.14165
      - 提供與嘗試相反的範例，希望語言模型答錯
        - 無效 Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? https://arxiv.org/abs/2202.12837
        - 對大的強的 有效 Larger language models do in-context learning differently https://arxiv.org/abs/2303.03846
      - 提供罕見語言的教科書，希望語言模型能翻譯
        - 有效 https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf

#2024.03.10 
- video: 
- slide: https://drive.google.com/file/d/1eVC4dx77Mba2_yMFe1_w4tXvIdSDTOCO/view?usp=sharing
- 3.拆解任務
- e.g. 分段寫長篇小說 Re3 arxiv
- e.g. 算數學有列式 CoT 
