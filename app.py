import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 모델 및 토크나이저 로딩
MODEL_NAME = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# 라벨 정의 (사용하는 모델에 따라 조정 필요)
label_map = {
    0: "부정",
    1: "중립",
    2: "긍정"
}

# 고객 발화만 추출
def extract_customer_utterances(text: str):
    pattern = r"고객\s*:\s*(.*)"
    return re.findall(pattern, text)

# 감정 예측 함수 (라벨 개수 동적 처리)
def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    label_id = torch.argmax(probs).item()
    return label_id, probs.squeeze().tolist()

# 예시 입력
sample_text = """
상담사 : 안녕하세요 고객님. 이번에 새로 출시된 보험 상품 안내드리려고 전화드렸습니다.
고객 : 아 네, 안그래도 보험이 필요했는데, 좋은 내용이겠죠?
상담사 : 건강보험인데요, 실손의료비와 입원비 보장이 포함되어 있고요...
상담사 : 월 납입금은 3만원이고 보장기간은 10년입니다.
고객 : 음...지금 바쁩니다. 전화하지 마세요.
고객 : 제 전화번호는 어떻게 아신거죠?.
"""

# 실행
customer_sentences = extract_customer_utterances(sample_text)

if not customer_sentences:
    print("고객 발화가 없습니다.")
else:
    # 첫 문장으로 클래스 개수 확인 및 라벨맵 생성
    _, first_scores = predict_sentiment(customer_sentences[0])
    num_classes = len(first_scores)
    # 라벨맵 자동 생성
    label_map = {i: f"클래스{i}" for i in range(num_classes)}
    sentiment_counts = {label_map[i]: 0 for i in range(num_classes)}
    sentiment_scores = {label_map[i]: 0.0 for i in range(num_classes)}
    # 클래스별 문장 저장용
    class_sentences = {label_map[i]: [] for i in range(num_classes)}

    for sentence in customer_sentences:
        label_id, scores = predict_sentiment(sentence)
        if len(scores) != num_classes:
            print(f"⚠️ 예측 결과 클래스 개수({len(scores)})가 {num_classes}이 아닙니다: {scores}")
            continue
        sentiment_counts[label_map[label_id]] += 1
        class_sentences[label_map[label_id]].append(sentence)
        for i in range(num_classes):
            sentiment_scores[label_map[i]] += scores[i]

    total = len(customer_sentences)
    avg_scores = {k: v / total for k, v in sentiment_scores.items()}
    overall_sentiment = max(avg_scores, key=avg_scores.get)
    print("🧾 고객 전체 감정 분석 결과\n" + "="*30)
    print(f"총 문장 수: {total}")
    print(f"문장별 감정 분포: {sentiment_counts}")
    print(f"평균 감정 확률: {['%s: %.2f' % (k, v) for k, v in avg_scores.items()]}")
    print(f"→ 전체적으로 '{overall_sentiment}' 감정이 우세합니다.\n")

    # 각 클래스별로 해당 문장(키워드) 출력
    print("클래스별 분류된 문장:")
    for label, sentences in class_sentences.items():
        print(f"{label} ({len(sentences)}개):")
        for s in sentences:
            print(f"  - {s}")
        print()
