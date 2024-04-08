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
    - 改變自己來強化模型 (improving inputs)
    - 訓練自己的模型 (improving parameters)
