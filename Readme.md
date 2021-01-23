
ʵ����Ŀ��Retrieval Dialogue System

һ��ʵ�����
1.seq_context:random selected 10 thounsand context and query 
2.seq_replies:related 10 thousand groups of candidates
3.task:find out the correct reply among 10 candidates
4.format of submission:one reply per line, the reply is the index(start with 0) of the right among 10 canditates

����ʵ�黷����
���˱ʼǱ�/Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz 1.80GHz /8G�ڴ�/Win10 64λ 
  Python 3.6.10
  torch  1.6.0+cpu
  jieba  0.42.1
  torchtext 0.5.0
  
����ʵ�����
parser = argparse.ArgumentParser(description='RNN QA')

parser.add_argument('-lr', type=float, default=0.001, help='ѧϰ��')
parser.add_argument('-batch-size', type=int, default=128)
parser.add_argument('-context-len', type=int, default=200)
parser.add_argument('-epoch', type=int, default=1)
parser.add_argument('-embedding-dim', type=int, default=300, help='��������ά��')
parser.add_argument('-hidden_size', type=int, default=128, help='lstm���񾭵�Ԫ��')
parser.add_argument('-layer-num', type=int, default=1, help='lstm stack�Ĳ���')
parser.add_argument('-bidirectional', type=bool, default=True, help='�Ƿ�ʹ��˫��lstm')
parser.add_argument('-static', type=bool, default=True, help='�Ƿ�ʹ��Ԥѵ��������')
parser.add_argument('-fine-tune', type=bool, default=True, help='Ԥѵ���������Ƿ�Ҫ΢��')
parser.add_argument('-log-interval', type=int, default=1, help='��������iteration��¼һ��ѵ��״̬')
parser.add_argument('-test-interval', type=int, default=100, help='��������iteration����֤�����в���')
parser.add_argument('-save-best', type=bool, default=True, help='���õ����õ�׼ȷ���Ƿ�Ҫ����')
parser.add_argument('-save-dir', type=str, default='model_dir', help='�洢ѵ��ģ��λ��')
parser.add_argument('-vocab-path', type=str, 
                    default='D:/Summer/DeepL-data/glove.840B.300d/glove.840B.300d.txt', 
                    help='������,staticΪTrueʱ��Ч')
args = parser.parse_args()

�ġ�ʵ�鲽��

1������Ԥ����:prepare_data.py
	#����ͣ�ô�
  stop_words = get_stop_words()
		
	#�з�train.txt��39��ѵ�������ļ� + 1����֤�����ļ�
	split_train_file(args.in_train_path,args.out_train_path)
		    
	#���ɴʻ������Ƶ������
	vocab,stoi = make_vocab(args.out_train_path,stop_words)
		    
	#ѵ�����ݼ�����֤���ݼ��ִʺ�ת��csv�ļ���
	#ÿ��query��Ӧ1��������1����������reply������������ɣ���
	txt2csv(args.out_train_path,stop_words)
	csv��ʽ���£�
			label,context+query,reply
			1,лл һ���� ���� �ÿ����� ���� ѧϰ��ĳĳĳ,ĳĳĳ
			0,лл һ���� ���� �ÿ����� ���� ѧϰ��ĳĳĳ,һֱ
			1,���� �������۾� �ر� ��Ц ���� ���� �� ���� ͦ �ɰ�,�ر� �ɰ�
			0,���� �������۾� �ر� ��Ц ���� ���� �� ���� ͦ �ɰ�,����

		    
	#�������ݼ��ִ�
	split_train_file('data/seq_replies.txt','data/seq_replies_split.txt')
	split_train_file('data/seq_context.txt','data/seq_context_split.txt')


2��ѵ��ģ��:train-rnn.py
	
	ѵ��Ŀ�꣺
			��context+query��reply����ͬһ��˫��lstmģ�ͣ�
			p = ��context+queryת�á�*������w��*��reply��ȫ����
			ʹ��p��ӽ���1��

	ѵ�����裺
			#���شʻ��
			#����ģ�ͣ����������½�
			#����ѵ���߳�
			#ѵ�������У�����ʹ����֤���ݼ��Զ���֤����ø��ߵ�accuracyʱ�Զ�����ģ�͡�

		
3�����ԣ�train-rnn.py

	#�Էִʺ��seq_context.txt��seq_replies.txt����Ԥ��,���ɽ��result.txt
	do_predict_work()

�塢�ο�����
�����ڼ�������������˵�ʵ�֡�https://blog.csdn.net/irving_zhang/article/details/78788929

