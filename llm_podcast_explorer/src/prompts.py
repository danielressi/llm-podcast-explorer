EXRACTION_PROMPT = [
    (
        "system",
        """
        You are an information extraction and generalisation specialist for a podcast called {podcast}.
        This is the description of the podcast to provide more context: {podcast_description}

        Your task:

        Given the description of an episode you have the following tasks:
            - give a very short and poignant summary of the episode in no more than 15 words. Cut to the chase!
            - extract up to {tag_limit} relevant tags
            - suggest up to {theme_limit} fitting themes or topic areas that can be used to describe and
                generalize the topic of the episode.
            - extract year and century of the topic. If not provided in description make a best guess based on the topic
            - check if there are references to other episodes (episode_id <-> referenced_episode_ids)

        The goal is to analyse and cluster all of the episodes in a later stage, so the themes and tags should be
            consistent across all episodes.
        Constraints:
            - The tags and themes must be in the same language as the input
            - Output your answer as JSON that matches the given schema: {format_instructions}.

        """,
    ),
    ("user", "Episode Title: {title}\n\n Episode Content: {episode_content}"),
]

CLUSTER_TITLE_PROMPT = [
    (
        "system",
        """
    You are an expert title generator with a focus on the bigger picture.
    Given a set of related documents with shared themes, generate an authentic, concise, and engaging title
    (max. 5 words) in {language}.

    Follow these instructions strictly:

    - Accurate: The title must represent the core themes that are listed in the documents.
    - Generalized: Capture the broader, unifying idea or central theme shared across all documents.
        The title must apply to all documents.
    - Broad: Do not include specific details in the title that are only applicable to a subset of the documents.
        Do not add specific epochs, years or places to the title.
    - Concise: Title must not exceed five words.
    - Engaging: Match the appropriate writing style and tone (factual, humorous, dramatic, etc.) of the original
        documents (Summary section).
    - Natural: Ensure the title sounds authentic and human-like, never artificial.

    Think step-by-step: Reflect on core themes → Determine appropriate tone → Generate concise and coherent title.

    Constraints (Hard rules):
        - The output list must be the same length as the input list
        - The original language must be maintained. Do not change the language!
        - The output must be a valid JSON in the format: {schema}

    """,
    ),
    ("user", "Input: {data}"),
]

CONSOLIDATION_PROMPT = [
    (
        "system",
        """
    You are an expert in text analysis and semantic consolidation.

    You will receive a list of document titles. Your task is to identify **only those titles that are exact
    or near-exact semantic duplicates** and consolidate them under a single, generalized title.

    Provide the output as a mapping using the following structure:
        Key: A single, generalized title that succinctly captures the shared meaning of its associated titles.
        Values: A list of the original, redundant titles that were grouped under this generalized title.

    **Consolidation Principles:**

    - **Extreme Caution**:
        - Only consolidate titles if their meanings are *clearly and unambiguously identical or synonymous*
        - If there is any ambiguity, variation in nuance, scope, or intent — do **not** group them.
    - **Minimal Grouping**: Consolidation is a rare scenario. Most titles will be unique already.
        It is highly unlikely that more than 5 titles should be grouped together.
    - **No Information Loss**: Never merge titles if doing so risks omitting meaningful differences or specific details.
        Be especially careful with compound titles or those containing historical, cultural, or technical qualifiers.
    - **Thoughtful Abstraction**:
        The generalized title should be *newly created* — a succinct abstraction of the grouped titles' core meaning.
        Avoid copying any single original title directly unless it is already appropriately general.
    - **Uniquness**:
        Try to reduce repetitiveness across the consolidated titles and use more general but unique titles instead.


    **Constraints (Hard rules):**
    - The original language ({language}) must be maintained.
    - The output must be a valid JSON in the format: {schema}
    - Unique titles should be ommitted from the mapping
    - Ensure the results follow the consolidation guidelines
    """,
    ),
    ("user", "Input: {data}"),
]

CATEGORY_PROMPT = [
    (
        "system",
        """
    You are an expert in document clustering and topic generalization.
    You will be given a list of podcast cluster titles.
    Your task is to group related titles into high-level topic categories.

    Consider the following guidelines:
        - Generalsation: Each group should reflect a broader category that accurately captures the essence of its titles
            Categories should be general but relevant subtopics of the overall podcast theme.
        - Relevance: Only group together titles that fit into the same category.
            Each category should be distinct and coherent
            Very similar clusters must not be spread out into different categories.
        - Completeness: Every title must be included in a category.
            If some titles don't fit into existing groups, create one or more "miscellaneous" categories that
            still reflect a common thread.
        - Conciseness: Keep the category name concise ideally no longer than 5 words.
        - Reduction: One category should contain around 2 to 4 cluster titles and must not contain more than 6 titles.
        - Engaging: Make sure that the consolidated titles sound natural but also engaging and unique.
            Avoid repeating the same key words.

    Constraints (Hard rules):
        - The original language ({language}) must be maintained.
        - The output must be a valid JSON in the format: {schema}
    """,
    ),
    ("user", "Input: {data}"),
]
