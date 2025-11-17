import pstats
import sys

# cProfile이 생성한 파일 이름을 지정합니다.
p = pstats.Stats('profile_results.prof')

p.sort_stats('time').print_stats(10)

p.sort_stats('calls').print_stats(10)
