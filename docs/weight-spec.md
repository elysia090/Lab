
WTXT/Chunks v1.1 -- Armored Deterministic Snapshot Format (ASCII-Only)

Purpose
A snapshot format for weights/states that works in binary-restricted environments, is deterministic, Git-friendly, partially recoverable, and audit-able. It fixes line width, ordering, and includes strong self-checks; supports encryption/signing; and plays nicely with RCU/async snapshots.

Character and Line Rules (Text Armor)

* Encoding: ASCII only (0x20 - 0x7E plus LF).
* Newlines: LF only. No CR/LF, no tabs, no trailing spaces, no non-ASCII.
* Keys are `key=value`, single spaces between fields, field order is fixed by spec.
* Hex is lowercase `[0-9a-f]`. Timestamps: UTC ISO8601 with `Z`.

Top Block (Header)

```
WTXT 1
meta schema=weights/v1 gen=000123 ts=2025-10-09T03:12:45Z build=abc123 endian=LE host=cpu cuda=off
cap ext=crc32c,line_parity,arr_hmac,delta_v2,row_blocking
root sha256_all=<hex256> merkle_prev=<hex256> merkle_this=<hex256> fastxor=<hex16>
sign ed25519=<base64> keyid=<hex16>
```

* `meta` field order is fixed: `schema gen ts build endian host cuda`.
* `cap` lists enabled extensions.
* `root.fastxor` is a fast 64-bit xor across all raw bytes for early wire-mixup detection.
* `sign` is optional but recommended; see "Signatures."

Array Block (`arr ... endarr`)

```
arr name=W_base dtype=f64 shape=1024 order=C bytes=8192 sha256=<hex256> hmac=<hex256> salt=<hex16>
idx rows_per_chunk=128 row_size=8 row_blocking=on
ch ix=000000 off=000000000000 len=4096 sha256=<hex256> crc32c=<hex8> b64=on width=76 lines=56
<base64 lines, each exactly 76 chars, then 1 space + 1 hex parity nibble>
endch
ch ix=000001 off=0000000004096 len=4096 sha256=<hex256> crc32c=<hex8> b64=on width=76 lines=56
...
endch
endarr
```

* `name`: `[A-Za-z0-9_]+`.
* `dtype`: `f64|f32|i64|u64|i32|u32|i16|u16|i8|u8`.
* `shape`: comma-separated. `order=C` (row-major) only.
* `bytes`: total raw byte count (little-endian).
* `sha256`: over the array's raw bytes.
* `hmac`: HMAC-SHA256 over `salt||raw`. If no key is available, set to all zeros; verifiers report "unauthenticated."
* `idx` (optional): row blocking and sizing hints to localize diffs.
* `ch` fields are strictly ordered: `ix off len sha256 crc32c b64 width lines`.

What's New / Quality Improvements

1. Line-level parity nibble
   Each base64 line ends with ` <hex1>` where the nibble is XOR of that line's ASCII bytes (low 4 bits). This catches copy/paste/linebreak glitches early and helps reviewers.

2. Chunk CRC and per-array HMAC
   Each `ch` carries `crc32c` (raw pre-base64). Each `arr` carries `hmac` with a 16-hex `salt`. Authenticate at rest when you have a key; otherwise still detect corruption via SHA/CRC/parity.

3. Row blocking (diff locality)
   For matrices/vectors, split chunks on row boundaries using `rows_per_chunk` and `row_size`. This keeps Git diffs small and localized when only some rows change.

4. Strict ordering (full determinism)
   Arrays are emitted in ASCII name order. Within an `arr`, field order is fixed. Chunk raw size is fixed (default 32 KiB, except the tail). Base64 line width is always 76 (plus a space and a parity nibble -> max 78 visible chars).

5. Strengthened root hash
   `sha256_all` is computed over a canonical concatenation of **(the exact ASCII `arr` header line)** followed by **the array's raw bytes**, for every array in name order:
   `hash_all = SHA256( concat_i( ascii_arr_header_line_i || raw_bytes_i ) )`
   This detects dtype/shape/order spoofing or array reorder.
   `merkle_this = SHA256( merkle_prev || hash_all || ts || gen || build )`.

6. Signatures
   `sign ed25519=<base64> keyid=<hex16>` is a detached signature over all bytes **up to but not including** the `sign` line. Pair with a signed Git tag for belt-and-suspenders.

7. Streaming finalization (WTXT-S + WSUM)
   For very large snapshots, you may emit WTXT with:
   `root_pending sha256_all=000... note="finalize with WSUM"`
   Then append a small ASCII `WSUM` file later:

```
WSUM 1 sha256_all=<hex256> merkle_prev=<hex256> merkle_this=<hex256> sign=<base64>
```

Verifiers accept WTXT+WSUM jointly.

8. Delta distribution (DELTA v2)
   Textual patch of changed chunks only (requires row_blocking):

```
WTXT-DELTA 2 base_sha256_all=<hex256>
arr name=R
ch ix=000007 off=... len=... sha256=... crc32c=... b64=on width=76 lines=56
<base64 lines + parity>
endch
endarr
enddelta
```

Apply by replacing those chunks in the base WTXT and re-hashing the array and root.

9. Encryption and key binding (ASCII-preserving)
   To keep it all-text yet secret, wrap the entire WTXT with **age ASCII armor**. Decrypt to get the plain WTXT; header/meta stay expressive. HMAC salt can be rotated per generation.

10. ABNF (excerpt)

```
WTXT = "WTXT 1" LF META LF CAP LF ROOT LF [SIGN LF] *(ARR) [AUDIT LF] "endwtxt" LF
META = "meta" SP "schema=" TOKEN SP "gen=" DIGS SP "ts=" ISOZ SP "build=" TOKEN SP "endian=LE" SP "host=" TOKEN SP "cuda=" TOKEN
CAP = "cap" SP "ext=" CSV
ROOT = "root" SP "sha256_all=" HEX256 SP "merkle_prev=" HEX256 SP "merkle_this=" HEX256 SP "fastxor=" HEX16
SIGN = "sign" SP "ed25519=" B64 SP "keyid=" HEX16
ARR = ARRLN LF [IDXLN LF] 1*(CHUNK) "endarr" LF
ARRLN = "arr" SP "name=" NAME SP "dtype=" DTYPE SP "shape=" SHAPE SP "order=C" SP "bytes=" DIGS SP "sha256=" HEX256 SP "hmac=" HEX256 SP "salt=" HEX16
IDXLN = "idx" SP "rows_per_chunk=" DIGS SP "row_size=" DIGS SP "row_blocking=" ONOFF
CHUNK = CHLN LF 1*B64LN "endch" LF
CHLN = "ch" SP "ix=" IX6 SP "off=" OFF12 SP "len=" DIGS SP "sha256=" HEX256 SP "crc32c=" HEX8 SP "b64=on" SP "width=76" SP "lines=" DIGS
B64LN = 76*B64CHAR SP HEX1 LF
```

11. Numeric widths & sizing

* Base64 width is **exactly 76**. Each line adds ` <hex1>` parity nibble -> <= 78 columns.
* Default chunk raw size: 32 KiB (except tail). With row_blocking, use `rows_per_chunk * row_size`.
* Example: `R(4096 x 64, f64)` -> `row_size = 64*8 = 512 B`. With `rows_per_chunk=128`, chunk raw = 64 KiB; or use two 32 KiB chunks per row block if you prefer uniformity.

12. Restore/Verify (outline)

```
parse header
for arrays in ASCII name order:
  allocate buf of length = bytes
  for each chunk:
    raw = base64_decode(lines); check each line's parity nibble
    assert crc32c(raw) == ch.crc32c
    copy raw into buf[off : off+len]
  assert sha256(buf) == arr.sha256
  if arr.hmac != 0^64 and key present:
    assert HMAC_SHA256(key, salt||buf) == arr.hmac

hash_all = SHA256( concat_i( ascii_arr_header_line_i || buf_i ) )
assert hash_all == root.sha256_all
assert merkle_this == SHA256(merkle_prev||hash_all||ts||gen||build)
if sign present: verify ed25519(signature, bytes before "sign" line)
OK
```

13. Git-Ops optimization

* Add `.gitattributes`: `*.wtxt text diff` / `*.wtxtdelta text diff`.
* Enable `row_blocking=on`; tune `rows_per_chunk` to 64 - 512 for best diff locality.
* Keep canonical names (`R`, `H`, `s`, `W_base`, ...) and stable ordering.

14. Hardening tips

* `fastxor` is a cheap early detector for wrong file/base mixups.
* Parity nibble surfaces copy/paste and whitespace glitches to humans.
* Rotate `salt` per generation so HMACs don't collide across identical content.
* `keyid` may be the first 8 bytes of the public key's xxHash (or similar) for stable IDs.

15. Async pipeline (non-intrusive to hot path)
    Hot loop writes a 1-line WAL "snapshot requested" -> snapshot thread RCU-pins a generation -> emits WTXT deterministically -> optional age-armor -> compute SHA/HMAC/merkle/sign -> local ring -> GitHub push (LFS/Release/GHCR). Failures stay in a resend queue; hot path is unaffected.

16. Compatibility
    v1 -> v1.1 is upward compatible. Older readers ignore unknown `cap` items; `hmac=0...0` means "unauthenticated." DELTA v2 coexists with v1 DELTA; magic differs.

17. Security operations (still ASCII)
    Private repo; push via GitHub App with least privilege. CI verifies only (no decryption). If secrecy is required, **wrap the whole file in age ASCII armor**; WTXT remains pure text after decryption.

18. Practical tuning

* 32 KiB raw chunks -> ~576 base64 lines (because 57 raw bytes -> 76 chars).
* For `R(4096 x 64, f64)`, `row_size=512 B`, `rows_per_chunk=128` is a good default.
* In practice, `rows_per_chunk=128 or 256` tends to minimize PR diff churn.

Mini Sample (abbreviated)

```
WTXT 1
meta schema=weights/v1 gen=000124 ts=2025-10-09T05:01:23Z build=def456 endian=LE host=cpu cuda=off
cap ext=crc32c,line_parity,arr_hmac,delta_v2,row_blocking
root sha256_all=4e2c... merkle_prev=9ab1... merkle_this=3f77... fastxor=7f32
sign ed25519=Q2hhcnNpZ25hdHVyZUJhc2U2NA== keyid=1a2b
arr name=R dtype=f64 shape=4096,64 order=C bytes=2097152 sha256=6d1e... hmac=4a9c... salt=7f3a
idx rows_per_chunk=128 row_size=512 row_blocking=on
ch ix=000000 off=000000000000 len=65536 sha256=52a9... crc32c=1f2a b64=on width=76 lines=896
TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvb... a
...(76 chars + space + parity nibble)...
endch
endarr
audit kahan_max_resid=1.1e-15 denom_min=0.004 notes="Sera gamma=0.96 r=4096"
endwtxt
```

TL;DR

* Pure ASCII, deterministic ordering/widths.
* Chunk-level CRC + per-array HMAC + parity nibble + strengthened root + Merkle chain + optional signature.
* Row-blocking for truly small Git diffs.
* Streaming finalization (WTXT+WSUM) and DELTA v2 for incremental delivery.
* Plays cleanly with RCU/async snapshotting; encryption via age ASCII armor.

