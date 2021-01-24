
实践项目三Retrieval Dialogue System

一、实验概述

	1.seq_context:random selected 10 thounsand context and query 
	2.seq_replies:related 10 thousand groups of candidates
	3.task:find out the correct reply among 10 candidates
	4.format of submission:one reply per line, the reply is the index(start with 0) of the right among 10 canditates

二、实验环境：

	个人笔记本/Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz 1.80GHz /8G内存/Win10 64位 
	  Python 3.6.10
	  torch  1.6.0+cpu
	  jieba  0.42.1
	  torchtext 0.5.0
  
三、实验参数

	parser.add_argument('-lr', type=float, default=0.001, help='学习率')
	parser.add_argument('-batch-size', type=int, default=128)
	parser.add_argument('-context-len', type=int, default=200)
	parser.add_argument('-epoch', type=int, default=1)
	parser.add_argument('-embedding-dim', type=int, default=300, help='词向量的维度')
	parser.add_argument('-hidden_size', type=int, default=128, help='lstm中神经单元数')
	parser.add_argument('-layer-num', type=int, default=1, help='lstm stack的层数')
	parser.add_argument('-bidirectional', type=bool, default=True, help='是否使用双向lstm')
	parser.add_argument('-static', type=bool, default=True, help='是否使用预训练词向量')
	parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调')
	parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
	parser.add_argument('-test-interval', type=int, default=100, help='经过多少iteration对验证集进行测试')
	parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
	parser.add_argument('-save-dir', type=str, default='model_dir', help='存储训练模型位置')
	parser.add_argument('-vocab-path', type=str, 
			    default='D:/Summer/DeepL-data/glove.840B.300d/glove.840B.300d.txt', 
			    help='词向量,static为True时生效')

四、实验步骤

1、数据预处理:prepare_data.py
	#加载停用词
  stop_words = get_stop_words()
		
	#切分train.txt，39个训练数据文件 + 1个验证数据文件
	split_train_file(args.in_train_path,args.out_train_path)
		    
	#生成词汇表，按词频倒排序
	vocab,stoi = make_vocab(args.out_train_path,stop_words)
		    
	#训练数据集和验证数据集分词后转成csv文件，
	#每个query对应1个正例、1个负例（从reply集合中随机生成）。
	txt2csv(args.out_train_path,stop_words)
	csv格式如下：
			label,context+query,reply
			1,谢谢 一切你 开心 好开心嗯 心里 学习，某某某,某某某
			0,谢谢 一切你 开心 好开心嗯 心里 学习，某某某,一直
			1,宿舍 厉害，眼睛 特别 搞笑 这土 不好 捏 觉得 挺 可爱,特别 可爱
			0,宿舍 厉害，眼睛 特别 搞笑 这土 不好 捏 觉得 挺 可爱,新闻

		    
	#测试数据集分词
	split_train_file('data/seq_replies.txt','data/seq_replies_split.txt')
	split_train_file('data/seq_context.txt','data/seq_context_split.txt')


2、训练模型:train-rnn.py
	
	训练目标：
			将context+query和reply经过同一个双向lstm模型，
			p = 【context+query转置】*【矩阵w】*【reply】全连接
			使得p最接近于1。

	训练步骤：
			#加载词汇表
			#加载模型，不存在则新建
			#启动训练线程
			#训练过程中，定期使用验证数据集自动验证，获得更高的accuracy时自动保存模型。

		
3、测试：train-rnn.py

	#对分词后的seq_context.txt，seq_replies.txt进行预测,生成结果result.txt
	do_predict_work()

五、参考文献

	《基于检索的聊天机器人的实现》https://blog.csdn.net/irving_zhang/article/details/78788929

