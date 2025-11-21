valgrind \
  --leak-check=full \
  --show-leak-kinds=all \
  --track-origins=yes \
  python3 tests/valgrind_check.py 2> valgrind.log
