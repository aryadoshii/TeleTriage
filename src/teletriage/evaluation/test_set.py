"""
Held-out evaluation set for TeleTriage — Phase 4.

CONSTRUCTION RULES (important — read before modifying):
  1. NO query may be a substring/normalised match of any key in
     data/sample_cache.json.  Cache keys: "high packet loss",
     "network congestion", "sip registration failing", "call drop",
     "handover failure", "weak signal", "high latency", "jitter on voice",
     "volte call failure", "throughput degradation", "lte attach failure",
     "low sinr", "dns resolution failure", "pdn connectivity failure",
     "roaming registration failure", "ims registration timeout",
     "enodeb unavailable", "paging failure", "authentication failure",
     "ip pool exhaustion", "cs fallback failure", "cell not visible to ue",
     "high prb utilization", "s1 setup failure", "voice quality degraded".

  2. NO query may duplicate a question already in data/sample_kb.jsonl
     (kb001–kb030).

  3. Queries SHOULD be semantically close to KB entries (so the retrieval
     tier is genuinely tested) but phrased differently enough that fuzzy
     cache matching won't fire.

  4. Reference answers are the gold standard for ROUGE/BERTScore.
     They are NOT the only correct answers — they are representative
     correct answers written by a subject-matter expert.

  5. `expected_tier` is a NON-BINDING hint for analysis.  The evaluator
     does NOT penalise a tier mismatch; it only uses the field to help
     interpret the per-tier breakdown.

Expected tier distribution:
  retrieval  ~12 / 20  — semantically close to KB entries
  generative  ~8 / 20  — novel queries with no KB coverage
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalItem:
    """One held-out evaluation example."""
    id: str
    query: str
    reference: str                   # gold-standard answer for metric scoring
    expected_tier: str = "unknown"   # "cache" | "retrieval" | "generative" | "unknown"
    tags: list[str] = field(default_factory=list)


EVAL_SET: list[EvalItem] = [
    # ── Retrieval-expected (semantically close to KB, different phrasing) ─────

    EvalItem(
        id="eval_001",
        query="UE keeps reselecting to 3G even when LTE coverage is adequate",
        reference=(
            "Idle-mode reselection to 3G when LTE is available is driven by "
            "cell selection/reselection parameters. Check: (1) SIB3/SIB5 — "
            "thresh-X-high and thresh-X-low for inter-RAT reselection. "
            "(2) RSRP of serving LTE cell — if below s-IntraSearchP the UE "
            "starts measuring neighbours. (3) IRAT priority configuration — "
            "if 3G cells have higher absolute priority they will be selected. "
            "(4) UE capability — is VoLTE supported? If not, UE may be "
            "configured to prefer UTRAN for CS domain. "
            "Review idle-mode measurement parameters in RRC SIB broadcast."
        ),
        expected_tier="retrieval",
        tags=["idle-mode", "reselection", "lte", "3g"],
    ),

    EvalItem(
        id="eval_002",
        query="Secondary cell addition for carrier aggregation fails silently",
        reference=(
            "When SCell addition fails without explicit failure: (1) Check "
            "RRCConnectionReconfiguration — is SCell configuration present and "
            "UE-NR/UE-EUTRA-Capability confirming CA band combination support? "
            "(2) Verify SCell is in the active neighbour relation list of PCell. "
            "(3) SCell RSRP threshold for addition (A4/A6 event configuration). "
            "(4) X2 interface between cells — SCell addition uses X2 for load "
            "information. (5) SCell scheduler load — may reject addition if PRB "
            "utilization is too high. Check eNB counters for CA addition attempts "
            "vs successes."
        ),
        expected_tier="retrieval",
        tags=["carrier-aggregation", "scell", "x2"],
    ),

    EvalItem(
        id="eval_003",
        query="PDCP SDU discard rate spiking on uplink for one cell sector",
        reference=(
            "High PDCP discard indicates packets are being dropped before "
            "transmission due to buffer overflow or discard timer expiry. "
            "Steps: (1) Check uplink SINR — poor radio quality causes HARQ "
            "retransmissions that back up the PDCP buffer. (2) UE transmit power "
            "headroom (PHR) — if UE is power-limited it cannot retransmit. "
            "(3) Uplink scheduler — is UE getting sufficient resource grants? "
            "(4) PDCP discard timer configuration — may be too aggressive. "
            "(5) Check for uplink interference source (PIM, external). "
            "Correlate with HARQ BLER and PHR statistics in the RRM counters."
        ),
        expected_tier="retrieval",
        tags=["pdcp", "uplink", "harq", "discard"],
    ),

    EvalItem(
        id="eval_004",
        query="Dedicated bearer for streaming video keeps being deactivated mid-session",
        reference=(
            "Non-GBR dedicated bearer teardown mid-session: (1) Check PCRF "
            "policy — is a Gx CCA-Update removing the bearer? TDF detection "
            "may be reclassifying traffic. (2) Inactivity timer on PGW for "
            "dedicated bearers — may be misconfigured. (3) SGW/PGW resource "
            "limits — bearer table exhaustion causes eviction. (4) UE-initiated "
            "bearer modification (QoS downgrade request). (5) Check ESM cause "
            "in Deactivate EPS Bearer Context Request. "
            "Trace the Gx interface for PCC rule changes correlated with bearer "
            "deactivation timestamps."
        ),
        expected_tier="retrieval",
        tags=["dedicated-bearer", "pcrf", "gx", "video"],
    ),

    EvalItem(
        id="eval_005",
        query="GTPv2-C signaling delay between MME and SGW causing session setup lag",
        reference=(
            "GTPv2-C (S11) delay causes PDN setup latency. Investigate: "
            "(1) Round-trip time on S11 path — run pings between MME and SGW "
            "loopback addresses. (2) SGW processing delay — check SGW CPU and "
            "message queue depth during busy hour. (3) Retransmissions on "
            "GTPv2-C — if Create Session Request is retransmitted, S11 RTT "
            "exceeds the T3-RESPONSE timer (default 3s). (4) IP routing "
            "asymmetry on the S11 VLAN. (5) SGW pool load balancing — one "
            "SGW may be overloaded. Use wireshark/tshark with GTP dissector "
            "to measure per-message RTT on the S11 interface."
        ),
        expected_tier="retrieval",
        tags=["gtpv2", "s11", "mme", "sgw", "latency"],
    ),

    EvalItem(
        id="eval_006",
        query="SRVCC handover from VoLTE to WCDMA failing during active call",
        reference=(
            "SRVCC (Single Radio Voice Call Continuity) failure: "
            "(1) Verify SVP (Sv) interface between MME and MSC-Server is up. "
            "(2) Check PS-to-CS handover trigger — B2 measurement event must "
            "fire with WCDMA threshold configured. (3) WCDMA cell must be in "
            "UE's active neighbour set (verify UTRAN frequency list in SIB6). "
            "(4) STN-SR (Session Transfer Number for SR-VCC) must be provisioned "
            "in HSS. (5) IMS ATU-STI (Access Transfer Update) must complete "
            "before WCDMA bearer is established. Check MME SRVCC counters — "
            "failure cause codes (e.g. target cell unavailable, Sv timeout) "
            "will narrow the root cause."
        ),
        expected_tier="retrieval",
        tags=["srvcc", "volte", "wcdma", "handover"],
    ),

    EvalItem(
        id="eval_007",
        query="Gx interface timeout causing all new bearer requests to be rejected",
        reference=(
            "Gx (PCRF interface) timeout blocks all dynamic PCC rule delivery. "
            "Steps: (1) Check Diameter routing — is the CCR-I reaching the PCRF? "
            "Verify Diameter realm routing on the DRA. (2) PCRF health — CPU, "
            "memory, SPR backend response time. (3) CCR timeout value on PGW — "
            "if shorter than PCRF processing time, spurious timeouts occur. "
            "(4) Check if PGW is configured with a default-rule fallback for "
            "Gx timeout (local policy). If not, all sessions requiring dynamic "
            "policy will fail. (5) Capture Diameter traffic on Gx — look for "
            "CCA response codes (5012 = Unable to Deliver). "
            "Immediate mitigation: enable local PCC fallback on PGW."
        ),
        expected_tier="retrieval",
        tags=["gx", "pcrf", "diameter", "bearer"],
    ),

    EvalItem(
        id="eval_008",
        query="NAS reject EMM cause 11 affecting subset of inbound roaming subscribers",
        reference=(
            "EMM cause 11 (PLMN not allowed) from MME during roaming: "
            "(1) Check VPLMN's allowed-PLMN list in MME — HPLMN PLMN ID may "
            "be missing. (2) OCS/PCRF roaming profile — subscription may not "
            "allow this service type for the HPLMN. (3) HSS ULA (Update Location "
            "Answer) flags — check 'PLMN-ODB' bit (Operator Determined Barring). "
            "(4) Roaming agreement active and reflected in systems? (5) Does "
            "it affect all IMSI ranges from that HPLMN or a subset (may indicate "
            "partial profile migration). Correlate MME EMM reject cause codes "
            "with specific HPLMN MCC-MNC pairs."
        ),
        expected_tier="retrieval",
        tags=["roaming", "emm", "nas", "plmn"],
    ),

    EvalItem(
        id="eval_009",
        query="eNB not generating handover decision despite UE sending A3 measurement reports",
        reference=(
            "When A3 events are received but no handover is initiated: "
            "(1) Verify A3 offset and hysteresis — if configured too aggressively "
            "the UE may trigger A3 but the source eNB disagrees. (2) Check "
            "neighbour relation — target cell must be in ANR table with valid "
            "PCI and EARFCN. (3) Target cell load rejection — X2 HO Request "
            "may be rejected by target due to high PRB utilization. (4) X2 "
            "interface health between source and target eNB — HO request may "
            "never reach the target. (5) Time-to-trigger (TTT) — is A3 event "
            "sustained for the full TTT before handover is attempted? "
            "Enable RRC trace on the source eNB to verify HO decision logic."
        ),
        expected_tier="retrieval",
        tags=["handover", "a3", "x2", "measurement-report"],
    ),

    EvalItem(
        id="eval_010",
        query="TCP sessions over SGi dropped after exactly 5 minutes of inactivity",
        reference=(
            "Exact 5-minute TCP idle teardown is a firewall or NAT timeout. "
            "(1) Check CGNAT / stateful firewall TCP idle timeout on SGi — "
            "default is often 300s (5 min). Extend or set application-specific "
            "timeout. (2) PGW session idle timer — not typical for TCP but "
            "confirm SGW/PGW inactive session timeout. (3) Load balancer "
            "session persistence timer. (4) Application keepalive — if TCP "
            "keepalive is not configured, the 5-minute idle hits. "
            "Fix: increase firewall TCP idle timer or enable SO_KEEPALIVE on "
            "the application. Capture RST origin (client vs server vs CGNAT) "
            "with tshark on SGi to confirm."
        ),
        expected_tier="retrieval",
        tags=["tcp", "sgi", "cgnat", "timeout"],
    ),

    EvalItem(
        id="eval_011",
        query="Uplink noise floor elevated on 3 contiguous PRBs suggesting narrowband interference",
        reference=(
            "Narrowband interference on specific PRBs is typically external RF "
            "or PIM. Steps: (1) Use spectrum analyzer or eNB uplink interference "
            "counters to confirm the affected frequency range. (2) PIM check: "
            "contiguous PRB pattern near downlink carriers is PIM signature — "
            "run PIM sweep on jumpers, connectors, and antennas. (3) External "
            "narrowband interferer (e.g. LTE-M device, IoT gateway) — use "
            "direction-finding / spectrum scan. (4) Check if interference is "
            "time-correlated (burst vs constant) to identify source type. "
            "(5) Inter-cell interference — verify ICIC is configured, check "
            "if neighbouring cell uses same PRBs at full power."
        ),
        expected_tier="retrieval",
        tags=["interference", "pim", "prb", "uplink", "rf"],
    ),

    EvalItem(
        id="eval_012",
        query="S1-AP reset PDU storm from one eNB causing MME instability",
        reference=(
            "Repeated S1-AP RESET from one eNB destabilizes MME UE context. "
            "(1) Identify cause IE in RESET — does eNB report an internal "
            "failure, radio link failure, or transport? (2) Check eNB hardware "
            "alarms — baseband restarts cause SCTP association reset. (3) "
            "Transport flap: SCTP heartbeat failures trigger RESET. Check "
            "interface counters on both sides. (4) Software issue on eNB — "
            "scheduled restart? Check OMC for maintenance windows. "
            "(5) MME-side mitigation: implement RESET rate limiting per eNB. "
            "Isolate the problematic eNB from the pool temporarily to protect "
            "the MME from context churn. Pull SCTP statistics and eNB alarm "
            "history from OMC."
        ),
        expected_tier="retrieval",
        tags=["s1ap", "reset", "mme", "sctp", "enb"],
    ),

    # ── Generative-expected (novel queries with limited KB coverage) ──────────

    EvalItem(
        id="eval_013",
        query="eSIM remote provisioning fails at LPA profile download with HTTP 403",
        reference=(
            "LPA (Local Profile Assistant) HTTP 403 during SM-DP+ profile "
            "download: (1) Activation code or confirmation code mismatch — "
            "re-check against SM-DP+ portal. (2) SMDP certificate not trusted "
            "by device — verify root CA chain. (3) Device EID not whitelisted "
            "in SM-DP+ operator profile. (4) Order state machine issue: "
            "profile may already be in 'downloaded' or 'released' state in "
            "SM-DP+ — check order status via M2M or consumer API. "
            "(5) Network connectivity: LPA uses HTTPS to SM-DP+ address — "
            "check DNS resolution and firewall rules on the device connection. "
            "Collect LPA logs (AT+CSIM or vendor diagnostic) and SM-DP+ "
            "server logs correlated by EID and timestamp."
        ),
        expected_tier="generative",
        tags=["esim", "lpa", "smdp", "provisioning"],
    ),

    EvalItem(
        id="eval_014",
        query="NB-IoT device cannot complete random access procedure despite strong RSRP",
        reference=(
            "NB-IoT RACH failure with good coverage: (1) NPRACH configuration — "
            "check subcarrier spacing, repetition levels, and periodicity in "
            "SIB2-NB. (2) Coverage level mismatch — device may be using wrong "
            "coverage enhancement level for its RSRP. (3) Cell capacity: if "
            "NPRACH resources are exhausted (congestion), RACH fails silently. "
            "(4) Device class: Cat-NB1 vs Cat-NB2 capabilities — verify device "
            "reports correct UE category. (5) Guard band vs in-band vs standalone "
            "deployment mode — device must be configured for the correct NB-IoT "
            "deployment type. Check eNB NB-IoT RACH statistics for preamble "
            "detection rate and coverage level distribution."
        ),
        expected_tier="generative",
        tags=["nb-iot", "rach", "coverage-enhancement"],
    ),

    EvalItem(
        id="eval_015",
        query="IPv4v6 PDN type request rejected by PGW, UE forced to IPv4 only",
        reference=(
            "PDN type IPv4v6 rejection (cause code 52 = single address bearers "
            "only, or 50 = PDN type IPv6 only allowed): (1) Check PGW APN "
            "configuration — APN may only have IPv4 pool, no IPv6 prefix "
            "delegation. (2) HSS subscription: subscribed PDN type may not "
            "include IPv4v6. (3) PGW license or platform limit — some vendors "
            "require specific license for dual-stack PDN. (4) UE behavior: "
            "after rejection with cause 52 the UE should retry with IPv4 — "
            "verify the retry is successful. (5) If IPv6 is required: provision "
            "IPv6 prefix pool on PGW and update APN profile in HSS."
        ),
        expected_tier="generative",
        tags=["ipv6", "pdn", "dual-stack", "pgw"],
    ),

    EvalItem(
        id="eval_016",
        query="Device receives ATTACH ACCEPT but immediately sends DETACH REQUEST",
        reference=(
            "ATTACH ACCEPT followed by immediate UE-initiated DETACH: "
            "(1) PDN connectivity failure post-attach — if default bearer "
            "establishment fails the device may detach. Check ESM cause in "
            "Activate Default EPS Bearer Context Request vs reject. "
            "(2) Device firmware issue — known bug in some chipsets where "
            "APN configuration mismatch triggers detach. (3) SIM application "
            "conflict — SIM STK applet may be overriding network selection. "
            "(4) Device detaches if IMS registration is required but fails "
            "immediately (IMS-only device). (5) Test with a different device "
            "and same SIM, and same device with a test SIM to isolate "
            "device vs subscription. Pull NAS trace from MME."
        ),
        expected_tier="generative",
        tags=["attach", "detach", "nas", "device"],
    ),

    EvalItem(
        id="eval_017",
        query="SMS over SGd interface not delivering messages for LTE-only subscribers without CSFB",
        reference=(
            "SGd-based SMS (SGs-less, direct MME-to-SMSC via SGd) delivery "
            "failure: (1) SGd (Diameter) interface between MME and SMS-GMSC "
            "must be up — check Diameter peer state. (2) Subscriber must have "
            "SMS-over-SGs or SMS-over-SGd activated in HSS — check subscription "
            "flags. (3) MSISDN routing in SMS-GMSC — for MT SMS, the SMSC must "
            "route via SGd path, not MAP/CAP legacy path. (4) P-CSCF and IMS "
            "not required for SGd SMS — but verify feature flag on MME. "
            "(5) For MO SMS: check that UE initiates NAS Uplink NAS Transport "
            "with SMS payload and MME routes to SGd. Trace both Diameter SGd "
            "and NAS on MME to find the break point."
        ),
        expected_tier="generative",
        tags=["sms", "sgd", "lte-sms", "mmse"],
    ),

    EvalItem(
        id="eval_018",
        query="Voice call audio drops out for first 2 seconds then recovers on every call",
        reference=(
            "Systematic 2-second audio gap at call start points to dedicated "
            "bearer establishment timing. (1) Measure time between SIP 200 OK "
            "and QCI 1 dedicated bearer activation — if > 1s, bearer setup is "
            "the culprit. (2) Jitter buffer pre-fill: some MGWs/UEs buffer "
            "before playing — check jitter buffer configuration. "
            "(3) Early media / ringback timing — is audio gate opening before "
            "RTP arrives? (4) RTCP negotiation delay on SBC — some SBCs wait "
            "for RTCP before opening audio. (5) SRTP key exchange — if DTLS "
            "or SDES handshake takes > 1s, media is delayed. Capture RTP at "
            "P-CSCF and UE simultaneously and measure first-packet timestamp."
        ),
        expected_tier="retrieval",
        tags=["volte", "audio", "bearer", "jitter-buffer"],
    ),

    EvalItem(
        id="eval_019",
        query="MIMO rank-2 throughput significantly worse than rank-1 for stationary UEs",
        reference=(
            "Rank-2 performing worse than rank-1 for stationary UEs indicates "
            "spatial channel issues: (1) Antenna cross-correlation too high — "
            "if physical antenna separation is insufficient, rank-2 spatial "
            "streams interfere. (2) Antenna port calibration mismatch — "
            "check RF calibration on both TX ports. (3) Precoding matrix "
            "indicator (PMI) feedback from UE — stationary UE should report "
            "stable PMI. If PMI is unstable, check UE feedback or downlink "
            "channel estimation. (4) CSI-RS configuration — incorrect CSI "
            "pilot power prevents accurate channel estimation. "
            "(5) UE receiver capability: some devices have poor rank-2 "
            "implementation. Test with a reference UE to isolate."
        ),
        expected_tier="retrieval",
        tags=["mimo", "rank", "precoding", "throughput"],
    ),

    EvalItem(
        id="eval_020",
        query="HSS returning Error-User-Unknown for valid IMSI during TAU procedure",
        reference=(
            "HSS Diameter error DIAMETER_ERROR_USER_UNKNOWN (5001) on S6a "
            "Update Location Request during TAU: (1) IMSI provisioned in HSS? "
            "Check directly in HSS subscriber DB — provisioning error is common. "
            "(2) IMSI format mismatch — HSS may store with/without leading zeros. "
            "(3) If moving between MMEs (inter-MME TAU) — old MME's IMSI "
            "de-registration may not have reached HSS. (4) HSS replication lag "
            "in geo-redundant setups — subscriber created on primary, TAU "
            "hit secondary before replication. (5) Subscription state: IMSI "
            "may be marked inactive in HSS (e.g. SIM swap in progress). "
            "Check HSS audit logs for the specific IMSI with full trace."
        ),
        expected_tier="retrieval",
        tags=["hss", "diameter", "tau", "imsi", "s6a"],
    ),
]


def get_eval_set() -> list[EvalItem]:
    """Return the full held-out evaluation set."""
    return list(EVAL_SET)


def get_eval_set_by_expected_tier(tier: str) -> list[EvalItem]:
    """Filter the eval set by expected tier hint."""
    return [item for item in EVAL_SET if item.expected_tier == tier]
