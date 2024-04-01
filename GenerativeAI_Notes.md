# 前言
快速筆記一下上課內容，以利日後Ctrl+F搜尋keywords和concept

# 2024.02.24【生成式AI導論 2024】第1講：生成式AI是什麼？ 
- https://www.youtube.com/watch?v=JGtqpQXfJis
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
