CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Werror -O2
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TEST_DIR = tests

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

EXEC = $(BIN_DIR)/mocd

# Default
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(OBJS) -o $@

# .c -> .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST_DIR)/test_community.o $(TEST_DIR)/test_optimization.o $(TEST_DIR)/test_utils.o
	$(CC) -o $(BIN_DIR)/test_community $(TEST_DIR)/test_community.o $(TEST_DIR)/test_optimization.o $(TEST_DIR)/test_utils.o

$(TEST_DIR)/test_community.o: $(TEST_DIR)/test_community.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_DIR)/test_optimization.o: $(TEST_DIR)/test_optimization.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_DIR)/test_utils.o: $(TEST_DIR)/test_utils.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)/*.o $(EXEC) $(BIN_DIR)/test_community

.PHONY: all clean test
