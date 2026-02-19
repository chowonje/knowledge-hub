#!/bin/bash
# Knowledge Hub 초기 설정 스크립트

set -e

echo "==================================="
echo "  Knowledge Hub 설정"
echo "==================================="

cd "$(dirname "$0")"

# 가상환경 생성
if [ ! -d "venv" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
source venv/bin/activate

# 패키지 설치
echo "패키지 설치 중..."
pip install -r requirements.txt

# 데이터 디렉토리 생성
mkdir -p data/papers data/chroma_db logs

# khub 명령어 심볼릭 링크 (선택사항)
echo ""
echo "==================================="
echo "  설정 완료!"
echo "==================================="
echo ""
echo "사용 방법:"
echo "  source venv/bin/activate"
echo "  python cli.py status"
echo "  python cli.py vault index"
echo "  python cli.py search '검색어'"
echo ""
echo "별칭 설정 (선택사항):"
echo "  echo 'alias khub=\"python $(pwd)/cli.py\"' >> ~/.zshrc"
echo "  source ~/.zshrc"
echo "  khub status"
