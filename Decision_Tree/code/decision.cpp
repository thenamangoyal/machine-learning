#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <utility>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <iomanip>
using namespace std;

map <int, map<int,int> > movie_dist; // Movie map to word and frequency
map <int, int > movie_rating;
set<int> word_list;
set<int> pos_movies;
set<int> neg_movies;

struct node{
	int word_attrib;
	int split_freq;
	int label; // -1: neg , 0: unassigned, 1: pos
	node* left;
	node* right;
};

node* id3(const set <int>& pos_example_movies, const set <int>& neg_example_movies,const set <int>& attrib_word_list);
node* early_id3(double threshold, const set <int>& pos_example_movies, const set <int>& neg_example_movies,const set <int>& attrib_word_list);
node* post_prune(node* root, node* curr, int level, int& level_to_prune, double& max_accur, const set <int>& validation_pos_reviews, const set <int>& validation_neg_reviews);
vector<node*> generate_random_forest(const set <int>& pos_example_movies, const set <int>& neg_example_movies,const set <int>& attrib_word_list, int n);

map<string, int> convert_word_split(const map<int,int>& split_count);
void word_used_as_split(node* root, map<int,int>& split_count);
void word_used_as_split_forest(const vector<node*>& forest, map<int,int>& split_count);
bool predict_rating(node* root, int movie);
bool predict_rating_forest(const vector<node*>& forest, int movie);
int no_of_nodes (node* root);
int count_terminal (node* root);
int height(node* root);
double accuracy(node* root, int start_index, const set <int>& pos_example_movies, const set <int>& neg_example_movies);
double accuracy_forest(const vector<node*>& forest, int start_index, const set <int>& pos_example_movies, const set <int>& neg_example_movies);

void help();

bool more_sec(const pair<int,int>& a, const pair<int,int>& b){
	return a.second > b.second;
}
int start_test_index = 0;

int main(int argc, char const *argv[])
{	
	srand (time(NULL));
	if (argc<2){
		cout<<"No input indices file specified"<<endl;
		help();
		return 1;
	}
	if (argc<3){
		cout<<"Expno not specified"<<endl;
		help();
		return 1;
	}

	if(atoi(argv[2]) > 5 || atoi(argv[2]) < 1){
		cout<<"Invalid expno"<<endl;
		help();
		return 1;
	}

	string temp_line;
	int count;

	if (atoi(argv[2]) == 1){
		string train_path = "train/labeledBow.feat";
		ifstream train_file(train_path.c_str());
		if (!train_file.good()){
			cout<<"Failed to open input file "<<train_path<<endl;
			help();
			return 1;
		}

		string test_path = "test/labeledBow.feat";
		ifstream test_file(test_path.c_str());
		if (!test_file.good()){
			cout<<"Failed to open input file "<<test_path<<endl;
			help();
			return 1;
		}

		string temp_line;
		int rating;

		vector<int> train_pos_reviews;
		vector<int> train_neg_reviews;

		int count;

		count = 0;
		while(getline(train_file,temp_line)){	
			stringstream input_stream(temp_line);
			if (input_stream>>rating){
				if (rating >= 7){
					train_pos_reviews.push_back(count);
				}
				else if (rating <= 4){
					train_neg_reviews.push_back(count);
				}
				count++;
			}
		}	

		// Randomly select positive and negative movies
		random_shuffle(train_pos_reviews.begin(), train_pos_reviews.end());
		random_shuffle(train_neg_reviews.begin(), train_neg_reviews.end());

		vector<int> sample_train_pos(train_pos_reviews.begin(), train_pos_reviews.begin()+500);
		vector<int> sample_train_neg(train_neg_reviews.begin(), train_neg_reviews.begin()+500);
		sort(sample_train_pos.begin(), sample_train_pos.end());
		sort(sample_train_neg.begin(), sample_train_neg.end());

		vector<int> sample_validation_pos(train_pos_reviews.begin()+500, train_pos_reviews.begin()+1000);
		vector<int> sample_validation_neg(train_neg_reviews.begin()+500, train_neg_reviews.begin()+1000);
		sort(sample_validation_pos.begin(), sample_validation_pos.end());
		sort(sample_validation_neg.begin(), sample_validation_neg.end());

		string selected_feature_path = argv[1];
		ofstream selected_feature(selected_feature_path.c_str(), ofstream::out | ofstream::trunc);

		train_file.close();
		train_file.open(train_path.c_str());

		for (int i=0; i<sample_train_pos.size(); i++){
			selected_feature<<sample_train_pos[i]<<endl;
		}
		selected_feature<<endl;

		for (int i=0; i<sample_train_neg.size(); i++){
			selected_feature<<sample_train_neg[i]<<endl;
		}
		selected_feature<<endl;

		for (int i=0; i<sample_validation_pos.size(); i++){
			selected_feature<<sample_validation_pos[i]<<endl;
		}
		selected_feature<<endl;

		for (int i=0; i<sample_validation_neg.size(); i++){
			selected_feature<<sample_validation_neg[i]<<endl;
		}
		selected_feature<<endl;

		vector<int> test_pos_reviews;
		vector<int> test_neg_reviews;

		count = 0;
		while(getline(test_file,temp_line)){	
			stringstream input_stream(temp_line);
			if (input_stream>>rating){
				if (rating >= 7){
					test_pos_reviews.push_back(count);
				}
				else if (rating <= 4){
					test_neg_reviews.push_back(count);
				}
				count++;
			}
		}

		random_shuffle(test_pos_reviews.begin(), test_pos_reviews.end());
		random_shuffle(test_neg_reviews.begin(), test_neg_reviews.end());

		vector<int> sample_test_pos(test_pos_reviews.begin(), test_pos_reviews.begin()+500);
		vector<int> sample_test_neg(test_neg_reviews.begin(), test_neg_reviews.begin()+500);
		sort(sample_test_pos.begin(), sample_test_pos.end());
		sort(sample_test_neg.begin(), sample_test_neg.end());

		test_file.close();
		test_file.open(test_path.c_str());

		for (int i=0; i<sample_test_pos.size(); i++){
			selected_feature<<sample_test_pos[i]<<endl;
		}
		selected_feature<<endl;

		test_file.close();
		test_file.open(test_path.c_str());
		for (int i=0; i<sample_test_neg.size(); i++){
			selected_feature<<sample_test_neg[i]<<endl;
		}

		selected_feature.close();
		train_file.close();
		test_file.close();

		cout<<"Selected indices saved to "<<argv[1]<<endl;
		cout<<endl;

	}
	else {

		ifstream indices_file(argv[1]);
		if (!indices_file.good()){
			cout<<"Failed to open input file "<<argv[1]<<endl;
			help();
			return 1;
		}
		set<int> validation_pos_reviews;
		set<int> validation_neg_reviews;

		set<int> test_pos_reviews;
		set<int> test_neg_reviews;

		map<int,int> complete_word_list;

		double temp_index;

		for (int i=0; i<500;i++){
			if(indices_file>>temp_index){
				pos_movies.insert(temp_index);
			}
			else{
				cout<<"Error reading indices files - Less than 3000 lines"<<endl;
				return 1;
			}
		}

		string train_path = "train/labeledBow.feat";
		string test_path = "test/labeledBow.feat";
		ifstream train_file;
		ifstream test_file;

		int temp_rating;
		int temp_word;
		char temp_char;
		int temp_freq;
		
		train_file.open(train_path.c_str());
		count = 0;
		getline(train_file,temp_line);
		for (set<int>::iterator i=pos_movies.begin(); i!=pos_movies.end(); ++i){
			while(count< *i){
				getline(train_file,temp_line);
				count++;
			}
			stringstream input_stream(temp_line);
			if (input_stream>>temp_rating){
				while(input_stream>>temp_word>>temp_char>>temp_freq){
					movie_dist[count].insert(make_pair<int,int>(temp_word,temp_freq));
					movie_rating[count] = temp_rating;
					if (complete_word_list.find(temp_word) != complete_word_list.end()){
						complete_word_list[temp_word]+=1;
					}
					else{
						complete_word_list[temp_word] = 1;
					}
				}
			}		
		}
		train_file.close();

		for (int i=0; i<500;i++){
			if(indices_file>>temp_index){			
				neg_movies.insert(temp_index);
			}
			else{
				cout<<"Error reading indices files - Less than 3000 lines"<<endl;
				return 1;
			}
		}

		train_file.open(train_path.c_str());
		count = 0;	
		getline(train_file,temp_line);
		for (set<int>::iterator i=neg_movies.begin(); i!=neg_movies.end(); ++i){		
			while(count< *i){
				getline(train_file,temp_line);
				count++;
			}
			stringstream input_stream(temp_line);
			if (input_stream>>temp_rating){
				while(input_stream>>temp_word>>temp_char>>temp_freq){
					movie_dist[count].insert(make_pair<int,int>(temp_word,temp_freq));
					movie_rating[count] = temp_rating;
					if (complete_word_list.find(temp_word) != complete_word_list.end()){
						complete_word_list[temp_word]++;
					}
					else{
						complete_word_list[temp_word] = 1;
					}
				}
			}
			
		}
		train_file.close();

		for (int i=0; i<500;i++){
			if(indices_file>>temp_index){
				validation_pos_reviews.insert(temp_index);
			}
			else{
				cout<<"Error reading indices files - Less than 3000 lines"<<endl;
				return 1;
			}
		}

		train_file.open(train_path.c_str());
		count = 0;	
		getline(train_file,temp_line);
		for (set<int>::iterator i=validation_pos_reviews.begin(); i!=validation_pos_reviews.end(); ++i){		
			while(count< *i){
				getline(train_file,temp_line);
				count++;
			}
			stringstream input_stream(temp_line);
			if (input_stream>>temp_rating){
				while(input_stream>>temp_word>>temp_char>>temp_freq){
					movie_dist[count].insert(make_pair<int,int>(temp_word,temp_freq));
					movie_rating[count] = temp_rating;
				}
			}
			
		}
		train_file.close();

		for (int i=0; i<500;i++){
			if(indices_file>>temp_index){
				validation_neg_reviews.insert(temp_index);
			}
			else{
				cout<<"Error reading indices files - Less than 3000 lines"<<endl;
				return 1;
			}
		}

		train_file.open(train_path.c_str());
		count = 0;	
		getline(train_file,temp_line);
		for (set<int>::iterator i=validation_neg_reviews.begin(); i!=validation_neg_reviews.end(); ++i){		
			while(count< *i){
				getline(train_file,temp_line);
				count++;
			}
			stringstream input_stream(temp_line);
			if (input_stream>>temp_rating){
				while(input_stream>>temp_word>>temp_char>>temp_freq){
					movie_dist[count].insert(make_pair<int,int>(temp_word,temp_freq));
					movie_rating[count] = temp_rating;
				}
			}
			
		}
		train_file.close();

		for (int i=0; i<500;i++){
			if(indices_file>>temp_index){
				test_pos_reviews.insert(temp_index);
			}
			else{
				cout<<"Error reading indices files - Less than 3000 lines"<<endl;
				return 1;
			}
		}

		// Map test movies starting from 2000 index
		start_test_index = pos_movies.size() + neg_movies.size() + validation_pos_reviews.size() + validation_neg_reviews.size();

		test_file.open(test_path.c_str());
		count = 0;	
		getline(test_file,temp_line);
		for (set<int>::iterator i=test_pos_reviews.begin(); i!=test_pos_reviews.end(); ++i){		
			while(count< *i){
				getline(test_file,temp_line);
				count++;
			}
			stringstream input_stream(temp_line);
			if (input_stream>>temp_rating){
				while(input_stream>>temp_word>>temp_char>>temp_freq){
					movie_dist[count+start_test_index].insert(make_pair<int,int>(temp_word,temp_freq));
					movie_rating[count+start_test_index] = temp_rating;
				}
			}
			
		}
		test_file.close();


		for (int i=0; i<500;i++){
			if(indices_file>>temp_index){
				test_neg_reviews.insert(temp_index);
			}
			else{
				cout<<"Error reading indices files - Less than 3000 lines"<<endl;
				return 1;
			}
		}

		test_file.open(test_path.c_str());
		count = 0;	
		getline(test_file,temp_line);
		for (set<int>::iterator i=test_neg_reviews.begin(); i!=test_neg_reviews.end(); ++i){		
			while(count< *i){
				getline(test_file,temp_line);
				count++;
			}
			stringstream input_stream(temp_line);
			if (input_stream>>temp_rating){
				while(input_stream>>temp_word>>temp_char>>temp_freq){
					movie_dist[count+start_test_index].insert(make_pair<int,int>(temp_word,temp_freq));
					movie_rating[count+start_test_index] = temp_rating;
				}
			}
			
		}
		test_file.close();

		indices_file.close();

		/*	Word list generate */
		set<int> whole_word_list;
		string word_file_path = "imdb.vocab";
		ifstream word_file(word_file_path.c_str());
		if (!word_file.good()){
			cout<<"Failed to open input file "<<word_file_path<<endl;
			return 1;
		}
		count = 0;
		string word;
		while(getline(word_file,temp_line)){
			stringstream input_stream(temp_line);
			if (input_stream>>word){
				whole_word_list.insert(count);
				count++;
			}
		}
		
		word_file.close();

		int no_word_to_take = 5000;
		if (argc>=4 && atoi(argv[3]) == 1) {
			string word_polarity_path = "imdbEr.txt";
			ifstream word_polarity_file(word_polarity_path.c_str());
			if (!word_polarity_file.good()){
				cout<<"Failed to open input file "<<word_polarity_path<<endl;
				return 1;
			}

			vector<pair<double,int> > polarity_whole_word_list;
			count = 0;
			double polarity;
			while (getline(word_polarity_file,temp_line)){
				stringstream input_stream(temp_line);
				if (input_stream>>polarity){
					polarity_whole_word_list.push_back(make_pair<double,int>(polarity, count));
					count++;
				}
			}
			word_polarity_file.close();

			sort(polarity_whole_word_list.begin(),polarity_whole_word_list.end());
			if (polarity_whole_word_list.size() < no_word_to_take/2){
				cout<<"Error word list too small"<<endl;
				return 1;
			}
			vector<pair<double,int> > top_word_list(polarity_whole_word_list.end()-no_word_to_take/2, polarity_whole_word_list.end());
			vector<pair<double,int> > bottom_word_list(polarity_whole_word_list.begin(), polarity_whole_word_list.begin()+no_word_to_take/2);
			
			for (int i=top_word_list.size()-1; i>=0; i--){
				word_list.insert(top_word_list[i].second);
			}
			
			for (int i=0; i<bottom_word_list.size(); i++){
				word_list.insert(bottom_word_list[i].second);
			}
		}
		else {
			vector<pair<int,int> > selected_word_list(complete_word_list.begin(), complete_word_list.end());
			
			sort(selected_word_list.begin(), selected_word_list.end(), more_sec);

			for (int i=0; i<no_word_to_take && i<selected_word_list.size(); i++){
				word_list.insert(selected_word_list[i].first);
			}
		}

		if (atoi(argv[2]) == 2){
		//ID3 with early stopping
			node* root = id3(pos_movies, neg_movies, word_list);
			cout<<"Original ID3"<<endl;

			cout<<"Training Accurary "<<accuracy(root, 0, pos_movies, neg_movies)<<endl;
			cout<<"Test Accurary "<<accuracy(root, start_test_index, test_pos_reviews, test_neg_reviews)<<endl;
			cout<<"Total Nodes "<<no_of_nodes(root)<<endl;
			cout<<"Terminal Nodes "<<count_terminal(root)<<endl;
			cout<<"Number of times an attribute is used as the splitting function"<<endl;

			map<int,int> count_word_splits;
			for (set<int>::iterator w_it = word_list.begin(); w_it != word_list.end(); ++w_it){
				count_word_splits[*w_it] = 0;
			}

			word_used_as_split(root, count_word_splits);

			map<string,int> string_word_splits = convert_word_split(count_word_splits);

			for (map<string,int>::const_iterator s_it=string_word_splits.begin(); s_it!=string_word_splits.end(); ++s_it){
				if ((*s_it).second){
					cout<<(*s_it).first<<" "<<(*s_it).second<<endl;	
				}
			}
			cout<<endl;

			cout<<"Effect of early stopping using threshold on information gain"<<endl;
			vector<double> threshold;
			threshold.push_back(0.1);
			threshold.push_back(0.07);
			threshold.push_back(0.05);
			threshold.push_back(0.03);
			threshold.push_back(0.01);

			for (int threshold_i=0; threshold_i<threshold.size(); threshold_i++) {
				node* root = early_id3(threshold[threshold_i], pos_movies, neg_movies, word_list);
				cout<<"Threshold "<<threshold[threshold_i]<<endl;

				cout<<"Training Accurary "<<accuracy(root, 0, pos_movies, neg_movies)<<endl;
				cout<<"Test Accurary "<<accuracy(root, start_test_index, test_pos_reviews, test_neg_reviews)<<endl;
				cout<<"Total Nodes "<<no_of_nodes(root)<<endl;
				cout<<"Terminal Nodes "<<count_terminal(root)<<endl;
				cout<<"Number of times an attribute is used as the splitting function"<<endl;

				count_word_splits.clear();
				for (set<int>::iterator w_it = word_list.begin(); w_it != word_list.end(); ++w_it){
					count_word_splits[*w_it] = 0;
				}

				word_used_as_split(root, count_word_splits);

				string_word_splits = convert_word_split(count_word_splits);

				for (map<string,int>::const_iterator s_it=string_word_splits.begin(); s_it!=string_word_splits.end(); ++s_it){
					if ((*s_it).second){
						cout<<(*s_it).first<<" "<<(*s_it).second<<endl;	
					}
				}
				cout<<endl;

			}

		}
		else if (atoi(argv[2]) == 3){
		//Noise
			cout<<"Effect of Noise in training data"<<endl;
			vector<double> noise_data;
			noise_data.push_back(0.5);
			noise_data.push_back(1);
			noise_data.push_back(5);
			noise_data.push_back(10);

			for (int noise_i=0; noise_i<noise_data.size(); noise_i++){
				set<int> set_noise_pos_movies;
				set<int> set_noise_neg_movies;

				int no_shuffle_elements = 1.0*(pos_movies.size()+neg_movies.size())*noise_data[noise_i]/100;

				if (no_shuffle_elements){
					int no_shuffle_pos_elements = rand()%no_shuffle_elements +1;
					no_shuffle_pos_elements = min(no_shuffle_pos_elements, int(pos_movies.size()));

					int no_shuffle_neg_elements = no_shuffle_elements - no_shuffle_pos_elements;
					no_shuffle_neg_elements = min(no_shuffle_neg_elements, int(neg_movies.size()));


					vector<int> index_set1;
					vector<int> index_set2;

					for (int i=0; i<pos_movies.size(); i++){
						index_set1.push_back(i);
					}
					for (int i=0; i<neg_movies.size(); i++){
						index_set2.push_back(i);
					}

					random_shuffle(index_set1.begin(), index_set1.end());
					random_shuffle(index_set2.begin(), index_set2.end());

					sort(index_set1.begin(), index_set1.begin()+no_shuffle_pos_elements);
					sort(index_set2.begin(), index_set2.begin()+no_shuffle_neg_elements);

					int index;

					count = 0;
					index = 0;
					for (set<int>::iterator p_it=pos_movies.begin(); p_it!=pos_movies.end(); ++p_it){
						if (index < no_shuffle_pos_elements && count==index_set1[index]){
							set_noise_neg_movies.insert(*p_it);
							index++;
						}
						else{
							set_noise_pos_movies.insert(*p_it);
						}
						count++;
					}

					count = 0;
					index = 0;
					for (set<int>::iterator n_it=neg_movies.begin(); n_it!=neg_movies.end(); ++n_it){
						if (index < no_shuffle_neg_elements && count==index_set2[index]){
							set_noise_pos_movies.insert(*n_it);
							index++;
						}
						else{
							set_noise_neg_movies.insert(*n_it);
						}
						count++;
					}

				}
				else{
					set_noise_pos_movies = pos_movies;
					set_noise_neg_movies = neg_movies;
				}
				
				node* root = id3(set_noise_pos_movies, set_noise_neg_movies, word_list);

				cout<<"Noise "<<noise_data[noise_i]<<endl;
				cout<<"Training Accurary "<<accuracy(root, 0, set_noise_pos_movies, set_noise_neg_movies)<<endl;
				cout<<"Test Accurary "<<accuracy(root, start_test_index, test_pos_reviews, test_neg_reviews)<<endl;
				cout<<"Total Nodes "<<no_of_nodes(root)<<endl;
				cout<<"Terminal Nodes "<<count_terminal(root)<<endl;
				cout<<endl;
			}
			
		}
		else if (atoi(argv[2]) == 4){
		//Post pruning
			cout<<"Effect of post pruning"<<endl;
			node* root = id3(pos_movies, neg_movies, word_list);
			cout<<"Before pruning"<<endl;

			cout<<"Training Accurary "<<accuracy(root, 0, pos_movies, neg_movies)<<endl;
			cout<<"Test Accurary "<<accuracy(root, start_test_index, test_pos_reviews, test_neg_reviews)<<endl;
			cout<<"Total Nodes "<<no_of_nodes(root)<<endl;
			cout<<"Terminal Nodes "<<count_terminal(root)<<endl;
			cout<<endl;

			count =0;
			double accur = accuracy(root, 0, validation_pos_reviews, validation_neg_reviews);
			int level_to_prune = -1;
			node* node_to_prune= post_prune(root, root, 0, level_to_prune, accur, validation_pos_reviews, validation_neg_reviews);
				
			while(node_to_prune){
				count++;
				node_to_prune->left = node_to_prune->right = NULL;
				
				cout<<"Pruning Stage "<<count<<endl;
			
				cout<<"Training Accurary "<<accuracy(root, 0, pos_movies, neg_movies)<<endl;
				cout<<"Test Accurary "<<accuracy(root, start_test_index, test_pos_reviews, test_neg_reviews)<<endl;
				cout<<"Total Nodes "<<no_of_nodes(root)<<endl;
				cout<<"Terminal Nodes "<<count_terminal(root)<<endl;
				cout<<endl;

				accur = accuracy(root, 0, validation_pos_reviews, validation_neg_reviews);
				level_to_prune = -1;
				node_to_prune= post_prune(root, root, 0, level_to_prune, accur, validation_pos_reviews, validation_neg_reviews);
			}			

		}
		else if (atoi(argv[2]) == 5){
		// Random forest
			cout<<"Effect of number of trees in the forest on train and test accuracies"<<endl;
			for (int i=1; i<= 128; i*=2){			
				vector<node*> forest = generate_random_forest(pos_movies, neg_movies, whole_word_list, i);
				cout<<i<<" "<<accuracy_forest(forest, 0, pos_movies, neg_movies)<<" "<<accuracy_forest(forest, start_test_index, test_pos_reviews, test_neg_reviews)<<endl;
			}
			cout<<endl;

		}

	}
	
	return 0;
}

vector<node*> generate_random_forest(const set <int>& pos_example_movies, const set <int>& neg_example_movies,const set <int>& attrib_word_list, int n){
	vector<node*> forest;
	vector<int> sample_word_list(attrib_word_list.begin(), attrib_word_list.end());
	int no_attrib = int(sqrt(sample_word_list.size()));
	for (int i =0; i<n; i++){
		random_shuffle(sample_word_list.begin(), sample_word_list.end());
		set<int> set_sample_word_list(sample_word_list.begin(), sample_word_list.begin() + no_attrib);
		
		node* root = id3(pos_example_movies, neg_example_movies, set_sample_word_list);

		forest.push_back(root);
	}
	return forest;
}


void help(){
	cout<<"Help"<<endl;
	cout<<"filename expno polarity"<<endl;
	cout<<"-- filename refers to the selected indices file"<<endl;
	cout<<"-- refers to expno from 1-5"<<endl;
	cout<<"-- polarity is optional. 0 (default) = use most frequent words; 1 = use highest and lowest polarity words"<<endl;
}

bool predict_rating_forest(const vector<node*>& forest, int movie){
	int pos_ratings = 0;
	int neg_ratings = 0;
	bool rating_predicted;
	for (int i =0; i<forest.size(); i++){
		rating_predicted = predict_rating(forest[i], movie);
		if (rating_predicted){
			pos_ratings++;
		}
		else{
			neg_ratings++;
		}
	}

	if (pos_ratings >= neg_ratings){
		return true;
	}
	return false;
}

map<string, int> convert_word_split(const map<int,int>& split_count){
	map<string, int> string_word_split;
	string word_file_path = "imdb.vocab";
	ifstream word_file(word_file_path.c_str());
	if (!word_file.good()){
		cout<<"Failed to open input file "<<word_file_path<<endl;
		return string_word_split;
	}
	int count = 0;
	string word;
	string temp_line;
	getline(word_file,temp_line);
	for (map<int,int>::const_iterator i=split_count.begin(); i!=split_count.end(); ++i){		
		while(count< (*i).first){
			getline(word_file,temp_line);
			count++;
		}
		stringstream input_stream(temp_line);
		if (input_stream>>word){
			string_word_split[word] = (*i).second;
		}
	}
	word_file.close();
	return string_word_split;
}


void word_used_as_split_forest(const vector<node*>& forest, map<int,int>& split_count){
	for (int i =0; i<forest.size(); i++){
		word_used_as_split(forest[i], split_count);
	}
}

void word_used_as_split(node* root, map<int,int>& split_count){
	if (root){
		if (root->word_attrib != -1){
			split_count[root->word_attrib]++;
		}
		if (root->left){
			word_used_as_split(root->left, split_count);
		}
		if (root->right){
			word_used_as_split(root->right, split_count);
		}
	}
}

double accuracy_forest(const vector<node*>& forest, int start_index, const set <int>& pos_example_movies, const set <int>& neg_example_movies){
	long long int wrong = 0;	
	for (set<int>::iterator it = pos_example_movies.begin(); it != pos_example_movies.end(); ++it){
		if (!predict_rating_forest(forest, *it + start_index)){
			wrong++;
		}
	}
	for (set<int>::iterator it = neg_example_movies.begin(); it != neg_example_movies.end(); ++it){
		if (predict_rating_forest(forest, *it + start_index)){
			wrong++;
		}
	}
	return 1.0*(pos_example_movies.size()+neg_example_movies.size() - wrong)/(pos_example_movies.size()+neg_example_movies.size());
}

double accuracy(node* root, int start_index, const set <int>& pos_example_movies, const set <int>& neg_example_movies){
	long long wrong = 0;
	for (set<int>::iterator it = pos_example_movies.begin(); it != pos_example_movies.end(); ++it){
		if (!predict_rating(root, *it + start_index)){
			wrong++;
		}
	}
	for (set<int>::iterator it = neg_example_movies.begin(); it != neg_example_movies.end(); ++it){
		if (predict_rating(root, *it + start_index)){
			wrong++;
		}
	}
	return 1.0*(pos_example_movies.size()+neg_example_movies.size() - wrong)/(pos_example_movies.size()+neg_example_movies.size());
}

bool predict_rating(node* root, int movie){
	if (!root->left && !root->right){
		if (root->label == 0){
			// error
			return true;
		}
		else if (root->label == 1){
			return true;
		}
		else {
			return false;
		} 
	}
	else if (!root->left){
		return predict_rating(root->right, movie);
	}
	else if (!root->right){
		return predict_rating(root->left, movie);
	}
	else{
		if (movie_dist[movie].find(root->word_attrib) != movie_dist[movie].end()){
			if (movie_dist[movie][root->word_attrib] <= root->split_freq){
				return predict_rating(root->left, movie);
			}
			else{
				return predict_rating(root->right, movie);
			}
		}
		else{
			return predict_rating(root->left, movie);
		}
	}
}

node* post_prune(node* root, node* curr, int level, int& level_to_prune, double& max_accur, const set <int>& validation_pos_reviews, const set <int>& validation_neg_reviews){
	node* left_node_to_prune = NULL;
	node* right_node_to_prune = NULL;
	int left_level_to_prune = -1;
	int right_level_to_prune = -1;

	if (curr->left){
		left_node_to_prune = post_prune(root, curr->left, level+1, left_level_to_prune, max_accur, validation_pos_reviews, validation_neg_reviews);
	}
	if (curr->right){
		right_node_to_prune = post_prune(root, curr->right, level+1, right_level_to_prune, max_accur, validation_pos_reviews, validation_neg_reviews);
	}

	if (curr && (curr->left || curr->right)){
		node* old_left = curr->left;
		node* old_right = curr->right;

		curr->left = NULL;
		curr->right = NULL;

		double new_accur = accuracy(root, 0, validation_pos_reviews, validation_neg_reviews);

		curr->left = old_left;
		curr->right = old_right;

		if (new_accur >= max_accur){
			max_accur = new_accur;
			level_to_prune = level;
			return curr;
		}
	}

	if (left_node_to_prune && right_node_to_prune){
		if (left_level_to_prune <= right_level_to_prune){
			level_to_prune = left_level_to_prune;
			return left_node_to_prune;
		}
		level_to_prune = right_level_to_prune;
		return right_node_to_prune;
	}
	else if (left_node_to_prune && !right_node_to_prune){
		level_to_prune = left_level_to_prune;
		return left_node_to_prune;
	}
	else if (!left_node_to_prune && right_node_to_prune){
		level_to_prune = right_level_to_prune;
		return right_node_to_prune;
	}
	return NULL;
}


node* id3(const set <int>& pos_example_movies, const set <int>& neg_example_movies,const set <int>& attrib_word_list){

	node* root = new node;

	root->label = 0;
	root->word_attrib = root->split_freq = -1;
	root->left = NULL;
	root->right = NULL;	
	if (pos_example_movies.size() >= neg_example_movies.size()){
		root->label = 1;
	}
	else{
		root->label = -1;
	}


	if (pos_example_movies.empty() == true && neg_example_movies.empty() == true){
	}
	else if (attrib_word_list.empty() == true){
	}
	else if (neg_example_movies.empty() == true){
	}
	else if (pos_example_movies.empty() == true){
	}
	else {
		//find max infomration gain
		double old_p = pos_example_movies.size();
		double old_n = neg_example_movies.size();

		double old_pp = old_p/(old_p+old_n);
		double old_pn = old_n/(old_p+old_n);

		double old_ent = -old_pp*log2(old_pp) - old_pn*log2(old_pn);

		int max_gain_word = -1;
		double info_gain = 0;
		double min_entropy = DBL_MAX;
		int split_value = -1;

		set<int> new_lpos_example_movies;
		set<int> new_lneg_example_movies;
		set<int> new_rpos_example_movies;
		set<int> new_rneg_example_movies;

		// iterate over all frequency values of a word
		for (set<int>::const_iterator word_it = attrib_word_list.begin(); word_it != attrib_word_list.end(); ++word_it){
			
			int curr_word = *word_it;
			double min_entropy_this_word = 	DBL_MAX;
			int split_this_word = -1;
			set<int> new_lpos_example_movies_this_word;
			set<int> new_lneg_example_movies_this_word;
			set<int> new_rpos_example_movies_this_word;
			set<int> new_rneg_example_movies_this_word;

			set<int> curr_lpos_example_movies_this_word;
			set<int> curr_lneg_example_movies_this_word;
			set<int> curr_rpos_example_movies_this_word;
			set<int> curr_rneg_example_movies_this_word;

			map<int, pair<set<int>, set<int> > > value_range;

			for (set<int>::const_iterator it = pos_example_movies.begin(); it != pos_example_movies.end(); ++it){
				if (movie_dist[*it].find(curr_word) != movie_dist[*it].end()){
					value_range[(movie_dist[*it][curr_word])].first.insert(*it);
				}				
				else{
					value_range[0].first.insert(*it);
				}
				curr_rpos_example_movies_this_word.insert(*it);
			}

			for (set<int>::const_iterator it = neg_example_movies.begin(); it != neg_example_movies.end(); ++it){
				if (movie_dist[*it].find(curr_word) != movie_dist[*it].end()){
					value_range[(movie_dist[*it][curr_word])].second.insert(*it);
				}
				else{
					value_range[0].second.insert(*it);
				}
				curr_rneg_example_movies_this_word.insert(*it);
			}

			if (value_range.size()>0){
				
				map<int, pair<int,int> > cumm_sum; //include current
				int pos_cumm_sum = 0;
				int neg_cumm_sum = 0;
				
				for (map<int, pair<set<int>, set<int> > >::const_iterator vit = value_range.begin(); vit != (value_range.end()); ++vit){
					pos_cumm_sum += (*vit).second.first.size();
					neg_cumm_sum += (*vit).second.second.size();			
					cumm_sum[(*vit).first] = make_pair<int,int> (pos_cumm_sum, neg_cumm_sum);
				}

				map<int, pair<int,int> > rev_cumm_sum; //exclude current
				pos_cumm_sum = 0;
				neg_cumm_sum = 0;
				for (map<int, pair<set<int>, set<int> > >::const_reverse_iterator vit = value_range.rbegin(); vit != (value_range.rend()); ++vit){
					rev_cumm_sum[(*vit).first] = make_pair<int,int> (pos_cumm_sum, neg_cumm_sum);
					pos_cumm_sum += (*vit).second.first.size();
					neg_cumm_sum += (*vit).second.second.size();	
				}

				for (map<int, pair<set<int>, set<int> > >::const_iterator vit = value_range.begin(); vit != --(value_range.end());){
					for (set<int>::iterator it2 = (*vit).second.first.begin(); it2 != (*vit).second.first.end(); ++it2){
						curr_lpos_example_movies_this_word.insert(*it2);
						curr_rpos_example_movies_this_word.erase(*it2);
					}
					for (set<int>::iterator it2 = (*vit).second.second.begin(); it2 != (*vit).second.second.end(); ++it2){
						curr_lneg_example_movies_this_word.insert(*it2);
						curr_rneg_example_movies_this_word.erase(*it2);
					}
					double lp = cumm_sum[(*vit).first].first;
					double ln = cumm_sum[(*vit).first].second;

					double lpp = lp/(lp+ln);
					double lpn = ln/(lp+ln);

					double lent = -lpp*log2(lpp) -lpn*log2(lpn);

					double rp = rev_cumm_sum[(*vit).first].first;
					double rn = rev_cumm_sum[(*vit).first].second;

					double rpp = rp/(rp+rn);
					double rpn = rn/(rp+rn);

					double rent = -rpp*log2(rpp) -rpn*log2(rpn);

					double entropy = ((lp+ln)*lent + (rp+rn)*rent)/(lp+ln+rp+rn);

					if (entropy < min_entropy_this_word){
						min_entropy_this_word = entropy;
						new_lpos_example_movies_this_word = curr_lpos_example_movies_this_word;
						new_lneg_example_movies_this_word = curr_lneg_example_movies_this_word;
						new_rpos_example_movies_this_word = curr_rpos_example_movies_this_word;
						new_rneg_example_movies_this_word = curr_rneg_example_movies_this_word;
						split_this_word = (*vit).first;
						++vit;
						split_this_word = (split_this_word + (*vit).first)/2;
					}
					else{
						++vit;
					}
				}				

			}

			if (min_entropy_this_word < min_entropy){
				info_gain = old_ent - min_entropy_this_word;
				max_gain_word = curr_word;
				min_entropy = min_entropy_this_word;
				split_value = split_this_word;
				new_lpos_example_movies = new_lpos_example_movies_this_word;
				new_lneg_example_movies = new_lneg_example_movies_this_word;
				new_rpos_example_movies = new_rpos_example_movies_this_word;
				new_rneg_example_movies = new_rneg_example_movies_this_word;

			}			

		}
		
		if ( ( (new_lpos_example_movies.empty() && new_lneg_example_movies.empty()) || (new_rpos_example_movies.empty() && new_rneg_example_movies.empty()))
			 || (max_gain_word==-1 || split_value==-1) ) {

		}
		else {

			root->word_attrib = max_gain_word;
			root->split_freq = split_value;
			set<int> new_attrib_word_list(attrib_word_list);
			new_attrib_word_list.erase(max_gain_word);
			root->left = id3(new_lpos_example_movies, new_lneg_example_movies, new_attrib_word_list);
			root->right = id3(new_rpos_example_movies, new_rneg_example_movies, new_attrib_word_list);
		}
		
	}
	return root;
}

node* early_id3(double threshold, const set <int>& pos_example_movies, const set <int>& neg_example_movies,const set <int>& attrib_word_list){

	node* root = new node;

	root->label = 0;
	root->word_attrib = root->split_freq = -1;
	root->left = NULL;
	root->right = NULL;
	if (pos_example_movies.size() >= neg_example_movies.size()){
		root->label = 1;
	}
	else{
		root->label = -1;
	}

	if (pos_example_movies.empty() == true && neg_example_movies.empty() == true){
	}
	else if (attrib_word_list.empty() == true){		
	}
	else if (neg_example_movies.empty() == true){
	}
	else if (pos_example_movies.empty() == true){
	}
	else {
		//find max infomration gain
		double old_p = pos_example_movies.size();
		double old_n = neg_example_movies.size();

		double old_pp = old_p/(old_p+old_n);
		double old_pn = old_n/(old_p+old_n);

		double old_ent = -old_pp*log2(old_pp) - old_pn*log2(old_pn);

		int max_gain_word = -1;
		double info_gain = 0;
		double min_entropy = DBL_MAX;
		int split_value = -1;

		set<int> new_lpos_example_movies;
		set<int> new_lneg_example_movies;
		set<int> new_rpos_example_movies;
		set<int> new_rneg_example_movies;

		// iterate over all frequency values of a word
		for (set<int>::const_iterator word_it = attrib_word_list.begin(); word_it != attrib_word_list.end(); ++word_it){
			
			int curr_word = *word_it;
			double min_entropy_this_word = 	DBL_MAX;
			int split_this_word = -1;
			set<int> new_lpos_example_movies_this_word;
			set<int> new_lneg_example_movies_this_word;
			set<int> new_rpos_example_movies_this_word;
			set<int> new_rneg_example_movies_this_word;

			set<int> curr_lpos_example_movies_this_word;
			set<int> curr_lneg_example_movies_this_word;
			set<int> curr_rpos_example_movies_this_word;
			set<int> curr_rneg_example_movies_this_word;

			map<int, pair<set<int>, set<int> > > value_range;

			for (set<int>::const_iterator it = pos_example_movies.begin(); it != pos_example_movies.end(); ++it){
				if (movie_dist[*it].find(curr_word) != movie_dist[*it].end()){
					value_range[(movie_dist[*it][curr_word])].first.insert(*it);
				}				
				else{
					value_range[0].first.insert(*it);
				}
				curr_rpos_example_movies_this_word.insert(*it);
			}

			for (set<int>::const_iterator it = neg_example_movies.begin(); it != neg_example_movies.end(); ++it){
				if (movie_dist[*it].find(curr_word) != movie_dist[*it].end()){
					value_range[(movie_dist[*it][curr_word])].second.insert(*it);
				}
				else{
					value_range[0].second.insert(*it);
				}
				curr_rneg_example_movies_this_word.insert(*it);
			}

			if (value_range.size()>0){
				
				map<int, pair<int,int> > cumm_sum; //include current
				int pos_cumm_sum = 0;
				int neg_cumm_sum = 0;
				
				for (map<int, pair<set<int>, set<int> > >::const_iterator vit = value_range.begin(); vit != (value_range.end()); ++vit){
					pos_cumm_sum += (*vit).second.first.size();
					neg_cumm_sum += (*vit).second.second.size();			
					cumm_sum[(*vit).first] = make_pair<int,int> (pos_cumm_sum, neg_cumm_sum);
				}

				map<int, pair<int,int> > rev_cumm_sum; //exclude current
				pos_cumm_sum = 0;
				neg_cumm_sum = 0;
				for (map<int, pair<set<int>, set<int> > >::const_reverse_iterator vit = value_range.rbegin(); vit != (value_range.rend()); ++vit){
					rev_cumm_sum[(*vit).first] = make_pair<int,int> (pos_cumm_sum, neg_cumm_sum);
					pos_cumm_sum += (*vit).second.first.size();
					neg_cumm_sum += (*vit).second.second.size();	
				}

				for (map<int, pair<set<int>, set<int> > >::const_iterator vit = value_range.begin(); vit != --(value_range.end());){
					for (set<int>::iterator it2 = (*vit).second.first.begin(); it2 != (*vit).second.first.end(); ++it2){
						curr_lpos_example_movies_this_word.insert(*it2);
						curr_rpos_example_movies_this_word.erase(*it2);
					}
					for (set<int>::iterator it2 = (*vit).second.second.begin(); it2 != (*vit).second.second.end(); ++it2){
						curr_lneg_example_movies_this_word.insert(*it2);
						curr_rneg_example_movies_this_word.erase(*it2);
					}
					double lp = cumm_sum[(*vit).first].first;
					double ln = cumm_sum[(*vit).first].second;

					double lpp = lp/(lp+ln);
					double lpn = ln/(lp+ln);

					double lent = -lpp*log2(lpp) -lpn*log2(lpn);

					double rp = rev_cumm_sum[(*vit).first].first;
					double rn = rev_cumm_sum[(*vit).first].second;

					double rpp = rp/(rp+rn);
					double rpn = rn/(rp+rn);

					double rent = -rpp*log2(rpp) -rpn*log2(rpn);

					double entropy = ((lp+ln)*lent + (rp+rn)*rent)/(lp+ln+rp+rn);

					if (entropy < min_entropy_this_word){
						min_entropy_this_word = entropy;
						new_lpos_example_movies_this_word = curr_lpos_example_movies_this_word;
						new_lneg_example_movies_this_word = curr_lneg_example_movies_this_word;
						new_rpos_example_movies_this_word = curr_rpos_example_movies_this_word;
						new_rneg_example_movies_this_word = curr_rneg_example_movies_this_word;
						split_this_word = (*vit).first;
						++vit;
						split_this_word = (split_this_word + (*vit).first)/2;
					}
					else{
						++vit;
					}
				}				

			}

			if (min_entropy_this_word < min_entropy){
				info_gain = old_ent - min_entropy_this_word;
				max_gain_word = curr_word;
				min_entropy = min_entropy_this_word;
				split_value = split_this_word;
				new_lpos_example_movies = new_lpos_example_movies_this_word;
				new_lneg_example_movies = new_lneg_example_movies_this_word;
				new_rpos_example_movies = new_rpos_example_movies_this_word;
				new_rneg_example_movies = new_rneg_example_movies_this_word;

			}

		}
		
		if ( (info_gain < threshold) || (
			( (new_lpos_example_movies.empty() && new_lneg_example_movies.empty()) || (new_rpos_example_movies.empty() && new_rneg_example_movies.empty()))
			 || (max_gain_word==-1 || split_value==-1)
			) ) {
		}
		else {

			root->word_attrib = max_gain_word;
			root->split_freq = split_value;
			set<int> new_attrib_word_list(attrib_word_list);
			new_attrib_word_list.erase(max_gain_word);
			root->left = early_id3(threshold, new_lpos_example_movies, new_lneg_example_movies, new_attrib_word_list);
			root->right = early_id3(threshold, new_rpos_example_movies, new_rneg_example_movies, new_attrib_word_list);
		}
		
	}
	return root;
}

int count_terminal (node* root){
	if (root){
		if (!root->left && !root->right){
			return 1;
		}
		return count_terminal(root->left) + count_terminal(root->right);
	}
	return 0;
}

int height(node* root){
	if (root){
		return 1 + max(height(root->left), height(root->right));
	}
	return 0;
}

int no_of_nodes (node* root){
	if (root){
		if (!root->left && !root->right){
			return 1;
		}
		return 1 + no_of_nodes(root->left) + no_of_nodes(root->right);
	}
	return 0;
}