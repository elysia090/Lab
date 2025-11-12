Verifiable End-to-End Network (VEEN) v0.0.1
	0.	Scope
A minimal network primitive where endpoints hold all semantics and cryptography; the hub provides ordering and reachability only. Every accepted message yields a signed receipt and is committed into an append-only Merkle Mountain Range (MMR) with logarithmic inclusion proofs. Authority is carried by portable capability tokens. Transport is abstract (HTTP, QUIC, NATS, file). The specification fixes byte-level encodings, signature coverage, and replay/audit artifacts for deterministic interoperability.
	1.	Notation
Byte concatenation is ||. u64be(n) is the 8-byte big-endian encoding of n. H is SHA-256. HKDF is HKDF-SHA256. AEAD is XChaCha20-Poly1305 (24-byte nonce). Sign/Verify are Ed25519. DH is X25519. HPKE refers to RFC 9180 base mode with KEM X25519HKDF-SHA256, KDF HKDF-SHA256, AEAD ChaCha20-Poly1305, and exporter interface. Domain-separated hashing: Ht(tag, x) = H( ascii(tag) || 0x00 || x ). All structured data use deterministic CBOR: maps with the exact field order listed in this spec; minimal-length unsigned integers; definite-length arrays and byte strings only; no indefinite-length items, no floating point, no CBOR tags. Fixed-length bstr are exact-size. Unknown map fields MUST be rejected. Inputs outside numeric ranges MUST be rejected.
	2.	Cryptographic profiles
A profile is an algorithm/parameter bundle.
profile = {
aead:“xchacha20poly1305”,
kdf:“hkdf-sha256”,
sig:“ed25519”,
dh:“x25519”,
hpke_suite:“X25519-HKDF-SHA256-CHACHA20POLY1305”,
epoch_sec:60,
pad_block:0,
mmr_hash:“sha256”
}
profile_id = Ht(“veen/profile”, CBOR(profile))
Every MSG carries profile_id. Receivers MAY reject unknown profile_id. Changing any profile field yields a different profile_id. Unless stated, H and mmr_hash are the same hash (sha256) in v0.0.1.
	3.	Keys and identities
Each client holds two long-term keypairs: id_sign (Ed25519) for signatures; id_dh (X25519) for prekey signing. Short-lived prekeys are X25519 public keys signed by id_sign. client_id in MSG is an Ed25519 public key verifying MSG.sig; it MUST rotate at least once per epoch if epoch_sec > 0, otherwise after at most M messages (RECOMMENDED M=256). Long-term identity, if needed, is referenced inside the encrypted payload via cap_ref and MUST NOT appear in plaintext fields. Hub verification keys (hub_pk) are distributed out of band and MAY be pinned by applications.
	4.	Streams and labels
A stream is identified by a 32-byte stream_id (e.g., H of an application-defined name). Routing confidentiality uses labels derived from a secret routing_key known only to stream members. For epoch E in Z, with E = floor(unix_time / epoch_sec) when epoch_sec > 0 and 0 otherwise,
label = Ht(“veen/label”, routing_key || stream_id || u64be(E))
The hub orders by label; it learns neither stream_id nor routing_key. Stream sequencing is global across epochs. When epoch_sec > 0, receivers SHOULD accept labels computed using E-1, E, or E+1 to tolerate moderate clock skew.
	5.	Wire objects
Three records appear on the wire. CBOR map field order is normative.

5.1 MSG (envelope visible to the hub)
Fields
ver: uint(=1)
profile_id: bstr(32)
label: bstr(32)
client_id: bstr(32)            ; Ed25519 public key, preferably epoch-ephemeral
client_seq: uint               ; per-(label,client_id) sequence, strictly increasing by 1
prev_ack: uint                 ; last observed stream_seq by the sender (gap hint)
auth_ref?: bstr(32)            ; Ht(“veen/cap”, CBOR(cap_token)) if hub admission applies
ct_hash: bstr(32)              ; H(ciphertext)
ciphertext: bstr               ; enc || hpke_ct_hdr || aead_ct_body
sig: bstr(64)                  ; Ed25519 over Ht(“veen/sig”, CBOR(MSG without sig))

Derived values
leaf_hash = Ht(“veen/leaf”, label || profile_id || ct_hash || client_id || u64be(client_seq))
msg_id = leaf_hash

Ciphertext formation (endpoint-only)
Let pkR be the receiver prekey. Produce HPKE base-mode sealing context for payload_hdr:
(enc, ctx_hdr) = HPKE.SealSetup(pkR)
hpke_ct_hdr = HPKE.Seal(ctx_hdr, aad=””, plaintext=CBOR(payload_hdr))
Derive a body AEAD key from HPKE exporter:
k_body = HPKE.Export(ctx_hdr, “veen/body-k”, 32)
nonce_body = Trunc_24( Ht(“veen/nonce”, label || u64be(prev_ack) || client_id || u64be(client_seq)) )
aead_ct_body = AEAD_Encrypt(k_body, nonce_body, aad=””, plaintext=body)
ciphertext = enc || hpke_ct_hdr || aead_ct_body
If profile.pad_block > 0, right-pad ciphertext with zero bytes to a multiple of pad_block before computing ct_hash. ct_hash = H(ciphertext).

5.2 RECEIPT (finality and commitment)
Fields
ver: uint(=1)
label: bstr(32)
stream_seq: uint               ; hub-assigned global sequence within the stream
leaf_hash: bstr(32)
mmr_root: bstr(32)
hub_ts: uint                   ; hub clock in seconds (informational)
hub_sig: bstr(64)              ; Ed25519 over Ht(“veen/sig”, CBOR(RECEIPT without hub_sig))

5.3 CHECKPOINT (epoch boundary anchor)
Fields
ver: uint(=1)
label_prev: bstr(32)
label_curr: bstr(32)
upto_seq: uint                 ; last stream_seq covered by this root
mmr_root: bstr(32)
epoch: uint                    ; E
hub_sig: bstr(64)
witness_sigs?: [ bstr(64) ]    ; optional co-signers
	6.	Payload header (encrypted, hub-blind)
payload_hdr is CBOR placed first inside ciphertext and AEAD-authenticated.
Fields
schema: bstr(32)               ; H of application schema bytes
parent_id?: bstr(32)           ; msg_id of parent (threads, RPC correlation)
att_root?: bstr(32)            ; Merkle root over attachment coids (see 10)
cap_ref?: bstr(32)             ; Ht(“veen/cap”, CBOR(cap_token))
expires_at?: uint              ; display-time hint only
	7.	Hub state and commitment
For each label the hub maintains S = (seq, peaks). seq in N0. peaks is an array of MMR peaks ordered by increasing tree size.
Append algorithm for leaf x:
7.1 seq := seq + 1
7.2 y := x; i := 0; while the i-th bit of seq is 0 do y := Ht(“veen/mmr-node”, peaks[i] || y); delete peaks[i]; i := i + 1; end; insert y as the new highest peak
7.3 If k = len(peaks) = 1 then mmr_root := Ht(“veen/mmr-root”, peaks[0]) else mmr_root := Ht(“veen/mmr-root”, peaks[0] || peaks[1] || … || peaks[k-1])
7.4 Emit RECEIPT(label, stream_seq=seq, leaf_hash=x, mmr_root, hub_ts, hub_sig)
The hub SHOULD verify MSG.sig, enforce size bounds, check plausibility of client_seq monotonicity, and admit only authorized writers when auth_ref is present. The hub MUST be agnostic to plaintext.
	8.	MMR inclusion proof
mmr_proof is CBOR
{ ver:1, leaf_hash:bstr(32), path:[ {dir:uint(0|1), sib:bstr(32)}… ], peaks_after:[ bstr(32) ] }
Verification: fold leaf_hash with path in order using Ht(“veen/mmr-node”, left||right) where dir=0 means sib is left, dir=1 means sib is right, to obtain a peak; then fold across peaks_after using Ht(“veen/mmr-root”, …). Equality with RECEIPT.mmr_root proves inclusion.
	9.	Client algorithms
9.1 Send
a) Build payload_hdr (schema and optional fields). Compute att_root if attachments exist.
b) Derive label for epoch E. Form ciphertext and ct_hash as in 5.1.
c) Prepare MSG; include auth_ref when hub admission is required; sign to obtain sig.
d) Submit MSG. Upon RECEIPT, verify hub_sig; check invariants I1..I9; update local MMR to RECEIPT.mmr_root; set prev_ack := RECEIPT.stream_seq.
e) Rekeying: after accepting RECEIPT with stream_seq = s, set rk_next = HKDF(rk, info=“veen/rk”||u64be(s)); derive direction keys HKDF(rk_next, info=“veen/send”) and HKDF(rk_next, info=“veen/recv”). Refresh HPKE at least once per epoch or every M messages.

9.2 Receive
For each pair (RECEIPT, MSG[, mmr_proof]):
a) Verify hub_sig. Check I1..I5, I7, I9. Ensure stream_seq is strictly increasing. Update local MMR to mmr_root.
b) Decrypt ciphertext to recover payload_hdr and body. If att_root exists, verify it by recomputing from attachments. Deliver to the application.
c) When epoch_sec > 0, accept labels derived from E-1, E, or E+1; prefer E based on local clock.
	10.	Attachments
Each attachment payload b is encrypted under a key derived from the HPKE exporter and attachment index i. Define k_att = HPKE.Export(ctx_hdr, “veen/att-k”||u64be(i), 32). Define n_att = Trunc_24( Ht(“veen/att-nonce”, msg_id || u64be(i)) ). Store ciphertext c = AEAD_Encrypt(k_att, n_att, aad=””, plaintext=b). Define coid = H(c). The set of coids is committed by att_root as the Merkle root using Ht(“veen/att-node”, left||right) and Ht(“veen/att-root”, peak1||…||peakk).
	11.	Capability tokens and admission
cap_token is CBOR
{ ver:1, issuer_pk:bstr(32), subject_pk:bstr(32), allow:{ stream_ids:[bstr(32)], ttl:uint, rate?:{per_sec:uint, burst:uint} }, sig_chain:[ bstr(64) ] }
Verification requires an acyclic signature chain from issuer to subject, unexpired ttl, and that stream_ids authorize the label’s stream. Hubs enforce stateless rate limits using a token bucket keyed by subject_pk or auth_ref. When enforced by the hub, MSG MUST carry auth_ref = Ht(“veen/cap”, CBOR(cap_token)). Admission requires an authorization step (Appendix A) that binds auth_ref to the validated cap_token at the hub prior to message submission.
	12.	Invariants (MUST hold per accepted (RECEIPT, MSG))
I1 H(ciphertext) = ct_hash
I2 leaf_hash = Ht(“veen/leaf”, label || profile_id || ct_hash || client_id || u64be(client_seq))
I3 mmr_root equals the root obtained by appending leaf_hash at stream_seq using section 7
I4 profile_id is supported by the client
I5 if payload_hdr.att_root exists then it equals the Merkle root of referenced coids
I6 prev_ack <= last observed stream_seq; equality implies no known gap
I7 capability constraints referenced via auth_ref and/or cap_ref hold at acceptance time
I8 within a label, the pair (client_id, client_seq) is unique across accepted messages
I9 client_seq increases by exactly 1 per client_id per label
	13.	Errors
E.SIG invalid MSG signature
E.SIZE size over limit
E.SEQ client_seq discontinuity (not previous+1)
E.CAP capability invalid or expired
E.AUTH missing or unknown auth_ref at hub
E.RATE rate exceeded
E.PROFILE unknown profile_id
E.DUP duplicate leaf_hash already accepted
E.TIME label epoch out of acceptance window
All errors are returned as CBOR {code:“E.*”, detail:text?}.
	14.	Security properties (informal)
Payload confidentiality holds against the hub by HPKE-sealed headers and AEAD-protected bodies. Authenticity and integrity hold by MSG.sig. Append-only history holds by the RECEIPT.mmr_root sequence. Equivocation is publicly provable by two RECEIPTs with identical (label, stream_seq) and distinct roots. Routing privacy follows from pseudorandom labels if routing_key remains secret and client_id rotates at least per epoch. Cross-stream replay is prevented by leaf_hash binding to label and profile_id. Nonce uniqueness follows from 5.1 and 10. Length-hiding is configurable via pad_block.
	15.	Portability set and file formats
Portable and sufficient for replay and audit: identity_card(pub), keystore.enc (encrypted id_sign_sk, id_dh_sk, prekey seed), routing_secret for the stream, receipts.cborseq, checkpoints.cborseq, payloads.cborseq (ciphertext and ct_hash), sync_state = {last_stream_seq, last_mmr_root}, capability tokens, and optional attachment contents addressed by coid. Files receipts.cborseq / checkpoints.cborseq / payloads.cborseq are CBOR Sequences (RFC 8742): concatenation of well-formed CBOR data items with no delimiters. Ratchet state is not portable; resynchronization derives fresh keys.
	16.	API surface (transport-agnostic)
submit: POST CBOR(MSG) -> CBOR(RECEIPT)
stream: GET label, from=stream_seq[, with_proof=bool] -> NDJSON of {RECEIPT, MSG, optional mmr_proof}
checkpoint_latest: GET label -> CHECKPOINT
checkpoint_range: GET epoch range -> sequence of CHECKPOINT
authorize: POST CBOR(cap_token) -> {auth_ref:bstr(32), expires_at:uint} (Appendix A)
report_equivocation: POST two RECEIPTs with the same (label, stream_seq) -> ok
Servers SHOULD reject undecodable CBOR or oversize payloads before any signature or HPKE work.
	17.	Complexity
Hub append is amortized O(1) time and O(log N) memory per label (peaks). Proof size and verification are O(log N). Client hot paths (send, receive, verify, decrypt) are O(1). Storage is linear in message count; MMR internal nodes can be compacted by retaining peaks only.
	18.	Interoperability discipline
Maps use the field order in this document; unknown fields are rejected in v0.0.1. Numeric fields use minimal unsigned encodings. Fixed-size bstr are exact length. peaks in 7.3 are ordered by increasing tree size. All domain-separated tags are prefixed with “veen/”. If profile.pad_block > 0, padding bytes are zeros and are included in ct_hash. Hubs MUST sign only after updating the MMR state and computing mmr_root. Clients MUST verify hub_sig before decryption.
	19.	Limits and bounds (normative defaults)
Implementations MUST enforce explicit limits; defaults below are RECOMMENDED and may be tightened by deployment policy:
MAX_MSG_BYTES = 1_048_576
MAX_BODY_BYTES = 1_048_320
MAX_HDR_BYTES = 16_384
MAX_PROOF_LEN = 64
MAX_CAP_CHAIN = 8
MAX_ATTACHMENTS_PER_MSG = 1024
CLOCK_SKEW_EPOCHS = 1 (accept E-1..E+1 when epoch_sec > 0)
These values are not encoded on the wire. Exceeding a bound MUST yield E.SIZE.
	20.	Minimal conformance vectors
Vector A (single writer, single message): epoch_sec=0; routing_key = 32 zero bytes; stream_id = H(“test”); client_id = Ed25519 public key; client_seq=1; prev_ack=0; payload_hdr with schema=H(“chat.v1”); empty body. Compute ciphertext per 5.1, ct_hash, MSG, RECEIPT, mmr_root. Verify I1..I9 and msg_id = leaf_hash.
Vector B (multi-writer): two distinct client_id alternating messages into the same label; each uses client_seq=1,2 per writer; VERIFY: (i) stream_seq strictly increases; (ii) (client_id,client_seq) uniqueness (I8); (iii) replay of an earlier MSG yields E.DUP.
Vector C (epoch roll): epoch_sec=60; send K messages in E; emit CHECKPOINT; rotate to E+1; first message under label_curr must advance from upto_seq+1; VERIFY: RECEIPT roots chain and CHECKPOINT ties labels.
	21.	Threat model notes (non-normative)
The hub is honest-but-curious: it may drop, delay, reorder across labels, rate-limit, or refuse admission, but it cannot forge hub_sig without hub_sk, nor undetectably rewrite accepted history. Compromised routing_key reveals label linkage but not payload plaintext. Compromised client_id affects linkability within its validity window. Pad-block hides coarse lengths at cost of bandwidth. Side-channel leakage (timing/size) remains out of scope beyond padding and admission gating.

Appendix A. Authorization flow (normative optional)
Purpose: allow hubs to enforce capability without inspecting encrypted payloads.
A.1 Endpoint: POST /authorize with CBOR(cap_token). Hub verifies cap_token (signature chain, ttl, allow.stream_ids) and returns {auth_ref:bstr(32), expires_at:uint}, where auth_ref = Ht(“veen/cap”, CBOR(cap_token)). Hub stores an authorization record keyed by auth_ref with expiry and rate limits.
A.2 Use: Client includes auth_ref in MSG.auth_ref. Hub admits MSG if an unexpired authorization record exists for auth_ref that authorizes the message’s label and rate bucket. Absence yields E.AUTH. Expiry yields E.CAP.
A.3 Revocation: Hub MAY revoke by deleting the record; subsequent messages fail with E.CAP.

Appendix B. Hub key distribution (non-normative)
Applications SHOULD pin hub_pk out of band (e.g., configuration) or rely on witness_sigs in CHECKPOINT for multi-party attestation. A pinned key change SHOULD be rolled with an overlapping window of dual-signed CHECKPOINTs.

Appendix C. CBOR skeletons (normative)
CBOR(MSG) fields appear exactly as listed in 5.1; CBOR(RECEIPT) as in 5.2; CBOR(CHECKPOINT) as in 5.3; CBOR(mmr_proof) as in 8; CBOR(cap_token) as in 11. All maps are encoded with keys in that order and contain no extra keys. All byte strings use definite length. All integers are nonnegative and minimally encoded.

Appendix D. Processing order (normative)
Hub receive path: decode CBOR -> size/bounds check -> verify MSG.sig -> authorize via auth_ref (if configured) -> append leaf to MMR and compute mmr_root -> sign RECEIPT -> respond.
Client receive path: verify hub_sig -> verify inclusion proof if supplied -> update local MMR -> decrypt -> deliver.
