Verifiable End-to-End Stream Network (VESN) v0.0.1
	0.	Scope
A minimal network primitive in which endpoints hold semantics and cryptography, while the network (the hub) provides ordering and reachability only. Every accepted message yields a signed receipt and becomes part of an append-only Merkle commitment with logarithmic inclusion proofs. Authority is carried by portable capability tokens. Transport is abstract (HTTP, QUIC, NATS, file).
	1.	Notation
Byte concatenation is ||. u64be(n) is the 8-byte big-endian encoding of n. All structured data are encoded as deterministic (canonical) CBOR; signatures always cover the exact CBOR byte string of the record with the signature field omitted. H is SHA-256, HKDF is HKDF-SHA256, AEAD is XChaCha20-Poly1305, Sign/Verify are Ed25519, DH is X25519. Define domain-separated hashing Ht(tag, x) = H( ascii(tag) || 0x00 || x ). All fixed-length fields are exact-size bstr.
	2.	Cryptographic profiles
A profile is an algorithm and parameter bundle. The default profile is
profile = { aead:“xchacha20poly1305”, kdf:“hkdf-sha256”, sig:“ed25519”, dh:“x25519”, epoch_sec:60, pad_block:0, mmr_hash:“sha256” }
profile_id = Ht(“vesn/profile”, CBOR(profile))
Every MSG carries profile_id. A receiver MAY reject unknown profile_id.
	3.	Keys and identities
Each client has two long-term keypairs: id_sign (Ed25519) for signatures, id_dh (X25519) for HPKE. Short-lived prekeys are X25519 public keys signed by id_sign. client_id in MSG is an Ed25519 public key used to verify MSG.sig; it MUST be rotated at least once per epoch if profile.epoch_sec > 0, otherwise after at most M messages (M is an implementation constant; RECOMMENDED M = 256). Long-term identity, if needed, is referenced inside the encrypted payload via cap_ref and MUST NOT be exposed in MSG.
	4.	Streams and labels
A stream is identified by a 32-byte stream_id (e.g., H of an application-defined name). Routing confidentiality is achieved by labels computed from a shared routing_key known to stream members only. For epoch E in Z, with E typically floor(unix_time / epoch_sec) when epoch_sec > 0 and 0 otherwise,
label = Ht(“vesn/label”, routing_key || stream_id || u64be(E))
The hub orders by label; it learns neither stream_id nor routing_key. Stream sequencing is global across epochs.
	5.	Wire objects
Three records appear on the wire.

5.1 MSG (message envelope; visible to the hub)
Fields
ver: uint(=1)
profile_id: bstr(32)
label: bstr(32)
client_id: bstr(32)       ; Ed25519 public key, preferably epoch-ephemeral
client_seq: uint          ; sender-local sequence for this stream, strictly increasing
prev_ack: uint            ; last observed stream_seq by the sender (for gap detection)
ct_hash: bstr(32)         ; H(ciphertext)
ciphertext: bstr          ; HPKE-sealed header + AEAD-protected body
sig: bstr(64)             ; Ed25519 over Ht(“vesn/sig”, CBOR(MSG without sig))

Derived values
leaf_hash = Ht(“vesn/leaf”, ct_hash || client_id || u64be(client_seq))
msg_id = leaf_hash

Ciphertext formation (endpoint-only)
The sender derives an HPKE context using its X25519 ephemeral and the receiver prekey bundle, then derives AEAD keys. The ciphertext is HPKE-sealed payload_hdr followed by AEAD(body). The hub cannot parse payload semantics.

5.2 RECEIPT (finality and commitment)
Fields
ver: uint(=1)
label: bstr(32)
stream_seq: uint          ; hub-assigned global sequence within the stream
leaf_hash: bstr(32)
mmr_root: bstr(32)
hub_ts: uint              ; hub clock in seconds
hub_sig: bstr(64)         ; Ed25519 over Ht(“vesn/sig”, CBOR(RECEIPT without hub_sig))

5.3 CHECKPOINT (epoch boundary anchor)
Fields
ver: uint(=1)
label_prev: bstr(32)
label_curr: bstr(32)
upto_seq: uint            ; last stream_seq covered by this root
mmr_root: bstr(32)
epoch: uint               ; E
hub_sig: bstr(64)
witness_sigs?: [ bstr(64) ]  ; optional co-signers
	6.	Payload header (encrypted, hub-blind)
payload_hdr is CBOR placed first inside ciphertext and AEAD-authenticated.
Fields
schema: bstr(32)          ; H of application schema bytes
parent_id?: bstr(32)      ; msg_id of parent (threads, RPC correlation)
att_root?: bstr(32)       ; Merkle root over attachment coids (see 10)
cap_ref?: bstr(32)        ; Ht(“vesn/cap”, CBOR(cap_token))
expires_at?: uint         ; display-time hint only
	7.	Hub state and commitment
For each label the hub maintains state S = (seq, peaks) with seq in N0, peaks a list of MMR peaks from smallest to largest tree size. Append algorithm upon accepting MSG with leaf_hash x
7.1 seq := seq + 1
7.2 y := x; i := 0; while the i-th bit of seq is 0 do y := Ht(“vesn/mmr-node”, peaks[i] || y); delete peaks[i]; i := i + 1; end; insert y as new highest peak
7.3 mmr_root := Ht(“vesn/mmr-root”, peaks[0] || peaks[1] || … || peaks[k-1]) with k = len(peaks); for k=1, mmr_root := Ht(“vesn/mmr-root”, peaks[0])
7.4 Emit RECEIPT(label, stream_seq=seq, leaf_hash=x, mmr_root, hub_ts, hub_sig)
The hub SHOULD verify MSG.sig, size bounds, plausibility of client_seq monotonicity, and capability admission (see 11). The hub MUST be agnostic to plaintext.
	8.	MMR inclusion proof
mmr_proof is CBOR
{ ver:1, leaf_hash:bstr(32), path:[ {dir:uint(0|1), sib:bstr(32)}… ], peaks_after:[ bstr(32) ] }
Verification folds leaf_hash with path in order (dir=0 means sib is left, dir=1 means sib is right) using Ht(“vesn/mmr-node”, left||right) to obtain a peak, then concatenates across peaks_after using Ht(“vesn/mmr-root”, …) to reconstruct mmr_root; equality with the RECEIPT mmr_root proves inclusion.
	9.	Client algorithms
9.1 Send
a) Build payload_hdr with schema and optional fields (parent_id, att_root, cap_ref, expires_at). For attachments, compute att_root as in 10.
b) HPKE-seal payload_hdr and AEAD-encrypt body to produce ciphertext; compute ct_hash = H(ciphertext).
c) Compute label from routing_key and epoch E; prepare MSG with fields, sign to produce sig.
d) Submit MSG; upon RECEIPT, verify hub_sig, check invariants I1..I4 and I6..I7 (section 12), and advance local MMR root to RECEIPT.mmr_root; set prev_ack := RECEIPT.stream_seq.
e) Rekeying: upon each accepted RECEIPT with stream_seq = s, set rk’ = HKDF(rk, info=“vesn/rk”||u64be(s)); derive direction keys as HKDF(rk’, info=“vesn/send”) and HKDF(rk’, info=“vesn/recv”); refresh HPKE at least once per epoch or every M messages.

9.2 Receive
For each streamed pair (RECEIPT, MSG[, mmr_proof])
a) Verify hub_sig. Check I1..I5, I7. Advance local MMR to mmr_root; ensure stream_seq is strictly increasing.
b) Decrypt ciphertext to obtain (payload_hdr, body). If att_root exists, verify it by recomputing from available attachments. Deliver to application.
	10.	Attachments
Each attachment payload b is encrypted under a key derived from the message HPKE context and an attachment index, then stored or transferred as bytes c. Define coid = H(c). The set of coids is committed by att_root as the Merkle root using Ht(“vesn/att-node”, left||right) and Ht(“vesn/att-root”, peak1||…||peakk). This prevents plaintext membership tests.
	11.	Capability tokens
cap_token is CBOR
{ ver:1, issuer_pk:bstr(32), subject_pk:bstr(32), allow:{ stream_ids:[bstr(32)], ttl:uint, rate?:{per_sec:uint, burst:uint} }, sig_chain:[ bstr(64) ] }
Verification requires an acyclic signature chain from issuer to subject, unexpired ttl, and that stream_ids authorise the label’s stream. Hubs enforce stateless rate limits using a token bucket keyed by subject_pk or cap_ref. cap_ref = Ht(“vesn/cap”, CBOR(cap_token)) may be placed in payload_hdr.
	12.	Invariants (MUST hold per accepted (RECEIPT, MSG))
I1 H(ciphertext) = ct_hash
I2 leaf_hash = Ht(“vesn/leaf”, ct_hash || client_id || u64be(client_seq))
I3 mmr_root equals the root obtained by appending leaf_hash at stream_seq using section 7
I4 profile_id is supported by the client
I5 if payload_hdr.att_root exists then it equals the Merkle root of referenced coids
I6 prev_ack <= last observed stream_seq; equality implies no known gap
I7 capability constraints referenced via cap_ref are satisfied at acceptance time
	13.	Errors
E.SIG invalid MSG signature
E.SIZE size over limit
E.SEQ client_seq discontinuity beyond tolerance
E.CAP capability invalid or expired
E.RATE rate exceeded
E.PROFILE unknown profile_id
E.DUP duplicate leaf_hash already accepted
All errors are returned as CBOR {code:“E.*”, detail:text?}.
	14.	Security properties (informal)
Under standard assumptions for Ed25519, X25519, HPKE, AEAD, and H, payload confidentiality holds against the hub; MSG authenticity and integrity hold by signature; append-only history holds by RECEIPT.mmr_root sequence; equivocation is publicly provable by two RECEIPTs with identical (label, stream_seq) and distinct roots; routing privacy follows from pseudorandom labels if routing_key remains secret and client_id is rotated per epoch.
	15.	Portability set
Portable and sufficient for replay and audit: identity_card(pub), keystore.enc (encrypted id_sign_sk, id_dh_sk, prekey seed), routing_secret for the stream, receipts.cborl, checkpoints.cborl, payloads.cborl (ciphertext and ct_hash), sync_state = {last_stream_seq, last_mmr_root}, capability tokens, and optional attachment contents addressed by coid. Ratchet state is not portable; resynchronization derives fresh keys.
	16.	API surface (transport-agnostic)
submit: POST CBOR(MSG) -> CBOR(RECEIPT)
stream: GET label, from=stream_seq -> NDJSON of {RECEIPT, MSG, optional mmr_proof}
checkpoint_latest: GET label -> CHECKPOINT
checkpoint_range: GET epoch range -> sequence of CHECKPOINT
report_equivocation: POST two RECEIPTs with the same (label, stream_seq) -> ok
	17.	Complexity
Hub append is O(1) time and memory amortized; proof size and verification are O(log N). Client hot paths (send, receive, verify, decrypt) are O(1). Storage is linear in message count; MMR internal nodes can be compacted by retaining peaks.
	18.	Interoperability discipline
Numeric fields use minimal-length unsigned CBOR. Peaks in 7.3 are ordered by increasing tree size. All domain-separated tags are prefixed with “vesn/”. profile.pad_block, if nonzero, right-pads ciphertext to a multiple of pad_block bytes before hashing; default is 0.
	19.	Compliance checklist
Hub implements section 7 and returns RECEIPT, maintains label-partitioned state, enforces 11 statelessly, serves 16. Client implements HPKE+AEAD, rotates client_id per section 3, enforces invariants 12, advances and verifies MMR, and exports the portability set 15.
	20.	Minimal conformance vectors (construction recipe)
Given fixed deterministic keys and a fixed profile, generate a single-message stream with epoch_sec=0, routing_key = 32 zero bytes, stream_id = H(“test”), client_id = Ed25519 public key, client_seq=1, prev_ack=0, payload_hdr with schema=H(“chat.v1”), empty body. Compute MSG, submit, compute RECEIPT and mmr_root, then verify I1..I7 and that msg_id equals leaf_hash. This vector SHALL be used to gate encoders, hash domains, and signature coverage.

