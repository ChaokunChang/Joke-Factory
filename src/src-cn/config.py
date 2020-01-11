class Config(object):
    model_prefix = "../checkpoints/jokes_cn"
    load_best_model = False
    # best_model_path  = "../checkpoints/best_train_model_cn"
    best_model_path = '/home/ubuntu/nlp-project/JokeGenerator/checkpoints/jokes_cn_70_2.3641119906947337'
    vocab_path = "/home/ubuntu/nlp-project/JokeGenerator/data/vocab/vocab1_cn.pickle"
    start_words = '有一个学生'
    prefix_words = None
    data_path = '/home/ubuntu/nlp-project/JokeGenerator/data/chinese_joke/duanzi.json'
    n_epochs = 200
    learning_rate = 1e-3
    batch_size = 32
    num_workers = 8 #加载数据时的线程数量
    num_layers = 2 #lstm层数
    embedding_dim = 256
    hidden_dim = 256
    tao = 0.8
    patience = 10
    use_gpu = True
    device = None

    save_every = 2 #每save_every个epoch存一次模型checkpoints,权重和诗,

    out_path = '../out_cn/'
    out_potery_path = '../out_cn/jokes.txt'
    len_limit = False
    max_seq_len = 200
    max_gen_len = 200 #最大生成长度
