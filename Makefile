PREFIX ?= /usr/local
BIN := target/release/llmtune

.PHONY: build release test check fmt clippy install clean

build:
	cargo build

release:
	cargo build --release

test:
	cargo test

fmt:
	cargo fmt

clippy:
	cargo clippy --all-targets -- -D warnings

check: fmt clippy test

install: release
	install -Dm755 $(BIN) $(DESTDIR)$(PREFIX)/bin/llmtune
	install -Dm644 profiles.toml $(DESTDIR)$(PREFIX)/share/llmtune/profiles.toml

clean:
	cargo clean
