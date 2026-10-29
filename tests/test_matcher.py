from langgraph_dynamic_tool_binding import KeywordToolMatcher, tokenize


def test_tokenize_splits_snake_and_camel_case():
    assert tokenize("purchaseOrder_line_item") == ["purchase", "order", "line", "item"]


def test_tokenize_drops_stopwords():
    assert tokenize("show me all the suppliers") == ["suppliers"]


def test_name_hits_rank_first(catalog):
    names = [t.name for t in KeywordToolMatcher().search("supplier update", catalog)]
    assert names[0] == "supplier_update"  # both tokens hit the name
    assert {f"supplier_{a}" for a in ["list", "get", "create", "delete"]} <= set(names)
    assert "customer_update" in names  # "update" alone still matches
    assert "customer_list" not in names


def test_partial_token_match(catalog):
    # "suppliers" contains "supplier"
    names = [t.name for t in KeywordToolMatcher().search("suppliers", catalog)]
    assert names and all(n.startswith("supplier_") for n in names)


def test_top_k_limits_results(catalog):
    assert len(KeywordToolMatcher(top_k=3).search("order customer", catalog)) == 3


def test_no_match_returns_empty(catalog):
    assert KeywordToolMatcher().search("weather", catalog) == []


def test_custom_stopwords(catalog):
    matcher = KeywordToolMatcher(stopwords={"order"})
    assert matcher.search("order", catalog) == []
