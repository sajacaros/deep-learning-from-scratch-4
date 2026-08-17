"""docs/questions.json 무결성 검사.

문항을 고친 뒤 실행해 형식이 깨지지 않았는지 확인한다.

    $ uv run python scripts/check_questions.py
"""

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DOCS = os.path.join(ROOT, 'docs')
QUESTIONS = os.path.join(DOCS, 'questions.json')

# 장 번호 -> 있어야 할 문항 수 (1~6장은 12개, 7~9장은 8개, 10장은 4개)
EXPECTED_COUNTS = {1: 12, 2: 12, 3: 12, 4: 12, 5: 12, 6: 12, 7: 8, 8: 8, 9: 8, 10: 4}
TOTAL = 100

VALID_TYPES = {'ox', 'blank', 'short', 'explain'}
REQUIRED_FIELDS = ('id', 'type', 'q', 'a')


def check(data):
    """검사해서 오류 메시지 목록을 돌려준다. 빈 목록이면 통과."""
    errors = []
    seen_ids = set()
    total = 0

    chapters = data.get('chapters')
    if not isinstance(chapters, list):
        return ['최상위에 chapters 배열이 없습니다']

    found = {ch.get('id') for ch in chapters}
    for missing in sorted(set(EXPECTED_COUNTS) - found):
        errors.append('{}장이 없습니다'.format(missing))
    for extra in sorted(found - set(EXPECTED_COUNTS), key=lambda x: (x is None, x)):
        errors.append('알 수 없는 장 번호입니다: {!r}'.format(extra))

    for ch in chapters:
        cid = ch.get('id')
        questions = ch.get('questions', [])
        total += len(questions)

        if not ch.get('title'):
            errors.append('{}장에 title이 없습니다'.format(cid))

        expected = EXPECTED_COUNTS.get(cid)
        if expected is not None and len(questions) != expected:
            errors.append('{}장 문항 수가 {}개입니다 (기대: {}개)'.format(
                cid, len(questions), expected))

        for i, q in enumerate(questions, 1):
            where = '{}장 {}번'.format(cid, i)

            for field in REQUIRED_FIELDS:
                if not q.get(field):
                    errors.append('{}: {} 필드가 비어 있습니다'.format(where, field))

            qid = q.get('id')
            if qid:
                if qid in seen_ids:
                    errors.append('{}: id가 중복됩니다 ({})'.format(where, qid))
                seen_ids.add(qid)

            qtype = q.get('type')
            if qtype and qtype not in VALID_TYPES:
                errors.append('{}: 알 수 없는 type입니다 ({}). 가능한 값: {}'.format(
                    where, qtype, ', '.join(sorted(VALID_TYPES))))

            fig = q.get('fig')
            if fig and not os.path.exists(os.path.join(DOCS, fig)):
                errors.append('{}: 그림 파일이 없습니다 (docs/{})'.format(where, fig))

    if total != TOTAL:
        errors.append('전체 문항 수가 {}개입니다 (기대: {}개)'.format(total, TOTAL))

    return errors


def summarize(data):
    lines = []
    for ch in data['chapters']:
        counts = {}
        for q in ch['questions']:
            counts[q['type']] = counts.get(q['type'], 0) + 1
        breakdown = '  '.join(
            '{} {}'.format(t, counts.get(t, 0)) for t in ('ox', 'blank', 'short', 'explain'))
        lines.append('  {:>2}장 {:<16} {:>3}문항   {}'.format(
            ch['id'], ch['title'], len(ch['questions']), breakdown))
    return '\n'.join(lines)


def main():
    try:
        with open(QUESTIONS, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print('questions.json 을 찾을 수 없습니다:', QUESTIONS)
        return 1
    except json.JSONDecodeError as e:
        print('JSON 형식이 잘못되었습니다: {} (줄 {}, 칸 {})'.format(e.msg, e.lineno, e.colno))
        return 1

    errors = check(data)

    print(summarize(data))
    print()

    if errors:
        print('문제 {}건을 찾았습니다.'.format(len(errors)))
        for e in errors:
            print('  -', e)
        return 1

    print('이상 없습니다. 총 {}문항.'.format(TOTAL))
    return 0


if __name__ == '__main__':
    sys.exit(main())
