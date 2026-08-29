from clean_text import clean_page


def test_drops_whole_line_headers():
    text = "OMH Official Policy Manual\nStaff may not install software.\nPage 3 of 7"
    assert clean_page(text) == "Staff may not install software."


def test_drops_dates_codes_and_rules():
    text = "3/14/16\nOM-500\n16-03\n_______\n1 of 2\nReal content."
    assert clean_page(text) == "Real content."


def test_keeps_line_that_merely_contains_a_header_word():
    text = "Introduction\nIntroduction to the policy manual follows."
    assert clean_page(text) == "Introduction to the policy manual follows."


def test_drops_blank_lines_and_strips_whitespace():
    text = "   Email is monitored.   \n\n\n   \nSo is browsing."
    assert clean_page(text) == "Email is monitored.\nSo is browsing."


def test_header_matching_is_case_insensitive():
    assert clean_page("date issued\nDATE ISSUED\nKept.") == "Kept."


def test_empty_page_yields_empty_string():
    assert clean_page("") == ""
