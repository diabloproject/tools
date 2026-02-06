TOOLS_ROOT := ../..
include $(TOOLS_ROOT)/build/Makefile.base

ifeq ($(UNAME_S),Darwin)
    SRC = ipc_mac.c
else ifeq ($(UNAME_S),Linux)
    SRC = ipc_lin.c
else
    SRC = ipc_win.c
endif

OBJS    = $(SRC:.c=.o)
TARGET  = libacho_ipc.a

.DEFAULT_GOAL := all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(AR) rcs $@ $^

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
