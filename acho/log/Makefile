TOOLS_ROOT := ../..
include $(TOOLS_ROOT)/build/Makefile.base

SRCS   = log.c
OBJS   = $(SRCS:.c=.o)
TARGET = libacho_log.a

.DEFAULT_GOAL := all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(AR) rcs $@ $(OBJS)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
