MARP := marp
PREFIX := docs

OBJECTS := $(PREFIX)/index.html 
OBJECTS += $(PREFIX)/lecture1.html 
OBJECTS += $(PREFIX)/lecture2.html 
# OBJECTS += $(PREFIX)/14.html
# OBJECTS += $(PREFIX)/15.html
# OBJECTS += $(PREFIX)/16.html

all: $(OBJECTS)

$(PREFIX)/%.html: %.md
	$(MARP) --html true $< -o $@ 
	cp -nr images/* $(PREFIX)/images/ &

.PHONY: clean

clean:
	rm -f $(PREFIX)/*.html