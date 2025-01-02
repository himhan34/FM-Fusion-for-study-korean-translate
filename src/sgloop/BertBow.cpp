#include "BertBow.h"

namespace fmfusion
{

    // 파일에서 바이트를 읽어오는 함수
    std::vector<char> get_the_bytes(const std::string &filename){
        std::ifstream input(filename, std::ios::binary); // 이진 모드로 파일 열기
        if(!input.is_open()){  // 파일 열기 실패 시
            throw std::runtime_error("Cannot open file " + filename + " for reading."); // 예외 처리
            return std::vector<char>(); // 빈 벡터 반환
        }

        // 파일에서 바이트를 읽어 벡터로 반환
        std::vector<char> bytes(
                (std::istreambuf_iterator<char>(input)),
                (std::istreambuf_iterator<char>()));
        input.close();  // 파일 닫기
        return bytes;  // 바이트 벡터 반환
    }

    // BertBow 클래스의 생성자
    BertBow::BertBow(const std::string &words_file, const std::string &word_features_file, bool cuda_device_):
        cuda_device(cuda_device_)  // CUDA 디바이스 사용 여부 설정
    {
        // 단어 사전과 단어 특징 파일을 로드
        if(load_word2int(words_file) && load_word_features(word_features_file))
        {
            N = word2int.size();  // 단어 수 저장
            assert(N==(word_features.size(0)-1));  // 단어 수와 특징 수가 일치하는지 확인
            load_success = true;  // 로드 성공 여부 설정
            if(cuda_device) word_features = word_features.to(torch::kCUDA);  // CUDA 디바이스로 데이터 이동
            std::cout<<"Load "<< N <<" words features on CUDA\n";  // 로드된 단어 특징 수 출력
            warm_up();  // 초기화
        }
        else{
            load_success = false;  // 로드 실패 시 설정
            std::cerr<<"Error: Bert Bag of Words Loaded Wrong!\n";  // 오류 메시지 출력
        }
    }

    // 주어진 단어들의 의미적 특징을 쿼리하는 함수
    bool BertBow::query_semantic_features(const std::vector<std::string> &words, 
                                    torch::Tensor &features)
    {
        int Q = words.size();  // 단어 수
        if(Q<1) return false;  // 단어 수가 1보다 적으면 실패
        float indices[Q];  // 단어 인덱스 배열

        // 각 단어에 대해 인덱스를 찾아 배열에 저장
        for(int i=0; i<Q; i++){
            if(word2int.find(words[i]) == word2int.end()){  // 단어가 사전에 없으면
                std::cerr<<"Error: word "<<words[i]<<" not found in the dictionary. Todo\n";  // 오류 메시지 출력
                indices[i] = 0;  // 기본값 0 할당
            }
            else{
                indices[i] = word2int[words[i]];  // 단어에 해당하는 인덱스 할당
            } 
        }
 
        indices_tensor = torch::from_blob(indices, {Q}, torch::kFloat32).toType(torch::kInt64);  // 인덱스 텐서 생성
        indices_tensor = indices_tensor.to(word_features.device());  // 텐서를 특징 텐서의 디바이스에 맞게 이동
        features = torch::index_select(word_features, 0, indices_tensor);  // 인덱스로 단어 특징 선택

        return true;  // 성공적으로 쿼리 완료
    }

    // 단어-인덱스 맵을 파일에서 로드하는 함수
    bool BertBow::load_word2int(const std::string &word2int_file)
    {
        // 단어-인덱스 맵 파일 열기
        std::ifstream file(word2int_file, std::ifstream::in);
        if (!file.is_open()){  // 파일 열기 실패 시
            std::cerr << "Error: cannot open file " << word2int_file << std::endl;  // 오류 메시지 출력
            return false;  // 실패 반환
        }
        else{
            std::string line;
            // 파일에서 한 줄씩 읽어 단어와 인덱스를 맵에 추가
            while (std::getline(file, line)){
                std::string label;
                int index;
                index = std::stoi(line.substr(0, line.find(".")));  // 인덱스 추출
                label = line.substr(line.find(".")+1, line.size()-line.find("."));  // 단어 추출

                word2int[label] = index;  // 단어와 인덱스 저장
            }
            return true;  // 성공적으로 로드
        }

    }

    // 단어 특징 파일을 로드하는 함수
    bool BertBow::load_word_features(const std::string &word_features_file)
    {

        std::vector<char> bytes = get_the_bytes(word_features_file);  // 파일에서 바이트 읽기
        if(bytes.empty()){  // 파일이 비었으면 실패
            return false;
        }
        else{
            // 파일에서 특징 로드
            torch::IValue ivalue = torch::pickle_load(bytes);
            word_features = ivalue.toTensor();  // 특징 텐서로 변환
            assert(torch::isnan(word_features).sum().item<int>()==0);  // NaN 값 체크

            torch::Tensor zero_features = torch::zeros({1, word_features.size(1)});  // 제로 특징 텐서 생성
            word_features = torch::cat({zero_features, word_features}, 0);  // 제로 특징을 첫 번째 행에 추가
            if(cuda_device) word_features = word_features.to(torch::kCUDA);  // CUDA 디바이스로 이동
            return true;  // 성공적으로 로드
        }

    }

    // 모델 초기화를 위한 워밍업 함수
    void BertBow::warm_up()
    {
        indices_tensor = torch::randint(0, N, {30}).toType(torch::kInt64);  // 임의의 인덱스 텐서 생성
        indices_tensor = indices_tensor.to(word_features.device());  // 특징 텐서의 디바이스에 맞게 이동
        torch::Tensor tmp_features = torch::index_select(word_features, 0, indices_tensor);  // 임의의 특징 쿼리 선택
        std::cout<<"Warm up BertBow with random quries \n";  // 워밍업 완료 메시지 출력
    }

}
