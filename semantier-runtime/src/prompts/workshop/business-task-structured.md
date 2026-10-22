请逐项核对这份虚构简历中的原文线索。输出 JSON，schema_version 为 workshop_business_candidate.v1。
checks 包含 education、experience、school_operations、systems、english、coordination、teaching_certificate 七个 criterion_id。
每项只有 criterion_id、status、source_span；status 为 SUPPORTED、MISSING 或 NEEDS_REVIEW。
存在原文线索时，source_span 为 governed_source 字符位置 {"start":整数,"end":整数}，不附带原文明文。
缺失信息必须标记 MISSING，source_span 为 null，不猜测或补齐。
证据位置只证明线索来自本次源，不证明岗位要求已满足。最终匹配和质量判断必须人工复核。
