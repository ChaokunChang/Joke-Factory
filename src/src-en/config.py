class Config(object):
    model_prefix = "../checkpoints/jokes"
    load_best_model = False
    best_model_path  = "../checkpoints/best_train_model"
    vocab_path = "/home/ubuntu/nlp-project/JokeGenerator/data/vocab/vocab10.pickle"
    start_words = 'It was a beautiful girl'
    prefix_words = None
    data_path = '/home/ubuntu/nlp-project/JokeGenerator/data/english_joke/reddit_jokes.json'
    n_epochs = 200
    learning_rate = 1e-3
    batch_size = 64
    num_workers = 8 #加载数据时的线程数量
    num_layers = 2 #lstm层数
    embedding_dim = 256
    hidden_dim = 256
    tao = 0.8
    patience = 10
    use_gpu = True
    device = None

    save_every = 2 #每save_every个epoch存一次模型checkpoints,权重和诗,

    out_path = '../out_en/'
    out_potery_path = '../out_en/jokes.txt'
    len_limit = False
    max_seq_len = 256
    max_gen_len = 200 #最大生成长度
