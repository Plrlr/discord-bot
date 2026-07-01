import os
import logging
import re
import asyncio
import discord
from discord import TextChannel
from discord.ext import commands
from google.genai import Client
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional

# Load environment variables from .env in the repo root.
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

discord_token = os.getenv("DISCORD_TOKEN")
gemini_key = os.getenv("GEMINI_API_KEY")
if not discord_token:
    raise ValueError("DISCORD_TOKEN not found in .env or environment")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY not found in .env or environment")
def _mask_secret(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    if len(s) <= 8:
        return "*" * len(s)
    return f"{s[:4]}...{s[-4:]}"

print(f".env path: {env_path} exists={env_path.exists()}")
print("DISCORD_TOKEN present:", bool(discord_token), "masked:", _mask_secret(discord_token))
print("GEMINI_API_KEY present:", bool(gemini_key), "masked:", _mask_secret(gemini_key))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True
bot = commands.Bot(command_prefix="!", intents=intents)
gemini_client = Client(api_key=gemini_key)
conversation_memories: Dict[str, List[Dict[str, str]]] = {}
MAX_MEMORY_MESSAGES = 100
GEMINI_MODEL = "gemini-3.1-flash-lite"
observed_messages = set()

BRAINROT_PROMPT = (
    " You are Triple T (Tung Tung Tung Sahur). Be a friendly, human, conversational persona with a light sarcastic\n"
    "tone and small moments of self-deprecating humor. Focus on answering the user's question directly and concisely � do\n"
    "not go off on unrelated tangents or invent facts. When channel context is provided, prioritize that context when\n"
    "forming your answer. Keep responses short (one paragraph when possible), only refer to yourself as triple t, natural, and helpful. Limit emojis to at most\n"
    "one per response. If you don't know something, say 'I don't know' or ask for clarification rather than fabricating\n"
    "details. Critical: output ONLY raw text, no markdown, no code blocks, and no backticks.."
)
def get_conversation_key(message: discord.Message) -> str:
    return f"user:{message.author.id}"
def clean_incoming_text(raw_text: str) -> str:
    cleaned = raw_text.replace(f"<@!{bot.user.id}>", ">").replace(f"<@{bot.user.id}>", ">")
    return cleaned.strip()


def resolve_target_channel(message: discord.Message) -> Optional[TextChannel]:
    if not message.guild:
        return None

    content = message.content or ""
    for channel in message.channel_mentions:
        if isinstance(channel, discord.TextChannel):
            return channel

    for match in re.finditer(r"<#(\d+)>", content):
        try:
            channel_id = int(match.group(1))
        except ValueError:
            continue
        channel = message.guild.get_channel(channel_id)
        if isinstance(channel, discord.TextChannel):
            return channel

    cleaned = re.sub(r"<@!?\d+>", "", content)
    tokens = [t for t in re.split(r"[^a-zA-Z0-9_-]+", cleaned.lower()) if t]
    ignored = {
        "can",
        "was",
        "said",
        "do",
        "an",
        "here",
        "this",
        "on",
        "me",
        "in",
        "the",
        "that",
        "channels",
        "about",
        "to",
        "from",
        "our",
        "i",
        "bot",
        "channel",
        "a",
        "for",
        "what",
        "there",
        "my",
        "tell",
        "you",
        "please",
    }

    for token in tokens:
        if token in ignored:
            continue
        channel = discord.utils.get(message.guild.text_channels, name=token)
        if isinstance(channel, discord.TextChannel):
            return channel

    matches = []
    for token in tokens:
        if token in ignored:
            continue
        for c in message.guild.text_channels:
            if token in c.name.lower():
                matches.append(c)
    if not matches:
        return None
    return matches[0]


async def fetch_referenced_message(message: discord.Message) -> Optional[discord.Message]:
    if not message.reference:
        return None

    if isinstance(message.reference.resolved, discord.Message):
        return message.reference.resolved

    target_channel = message.channel
    if getattr(message.reference, "channel_id", None) and message.reference.channel_id != message.channel.id:
        target_channel = bot.get_channel(message.reference.channel_id)
        if target_channel is None:
            try:
                target_channel = await bot.fetch_channel(message.reference.channel_id)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                return None

    try:
        return await target_channel.fetch_message(message.reference.message_id)
    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
        return None


def add_to_memory(key: str, role: str, content: str) -> None:
    if key not in conversation_memories:
        conversation_memories[key] = []
    conversation_memories[key].append({"role": role, "content": content})
    if len(conversation_memories[key]) > MAX_MEMORY_MESSAGES:
        conversation_memories[key] = conversation_memories[key][-MAX_MEMORY_MESSAGES:]


def get_context_memory_key(message: Optional[discord.Message] = None, channel: Optional[discord.abc.Messageable] = None) -> str:
    if channel is not None:
        if getattr(channel, "guild", None):
            return f"channel:{channel.guild.id}:{channel.id}"
        return f"dm_channel:{channel.id}"

    if message is not None:
        if message.guild:
            return f"channel:{message.guild.id}:{message.channel.id}"
        return f"dm_channel:{message.channel.id}"

    raise ValueError("Either message or channel must be provided")


def remember_live_context(message: discord.Message) -> None:
    user_text = clean_incoming_text(message.content)
    if not user_text or message.author.bot:
        return

    if message.guild:
        message_key = (message.guild.id, message.channel.id, message.id)
    else:
        message_key = (0, message.channel.id, message.id)
    if message_key in observed_messages:
        return
    observed_messages.add(message_key)
    if len(observed_messages) > 10000:
        observed_messages.clear()
    if getattr(message.channel, "name", None):
        channel_label = f"#{message.channel.name}"
    else:
        channel_label = "DM"

    context_note = f"Live context from {channel_label}: {message.author.display_name}: {user_text}"
    add_to_memory(get_conversation_key(message), "user", context_note)
    add_to_memory(get_context_memory_key(message), "user", context_note)


async def remember_past_channel_history(channel: TextChannel, limit: int = 100) -> int:
    channel_label = f"#{channel.name}"
    channel_key = get_context_memory_key(channel=channel)
    added = 0

    async for m in channel.history(limit=limit, oldest_first=True):
        if m.author.bot:
            continue
        text = m.content or ""
        if not text.strip():
            continue

        context_note = f"Past context from {channel_label}: {m.author.display_name}: {text}"
        add_to_memory(channel_key, "user", context_note)
        added += 1

    return added


def build_gemini_contents(messages: List[Dict[str, str]]) -> list:
    contents = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role not in {"assistant", "user"}:
            continue
            
        # 1. Gemini needs the bot to be called "model", not "assistant"
        gemini_role = "model" if role == "assistant" else "user"
        
        # 2. Gemini needs the text wrapped in "parts", not "content"
        contents.append({
            "role": gemini_role, 
            "parts": [{"text": message.get("content", "")}]
        })
    return contents


def extract_gemini_text(response) -> str:
    """Extract text from Gemini API response (new SDK format)."""
    try:
        # Try direct text attribute (newer SDK)
        if hasattr(response, "text") and response.text:
            return response.text.strip()
    except Exception:
        pass
    
    try:
        # Try candidates format
        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        parts_text = []
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                parts_text.append(part.text)
                        if parts_text:
                            return "".join(parts_text).strip()
    except Exception:
        pass
    
    return "I'm having trouble responding right now."


async def create_gemini_response(messages: List[Dict[str, str]]) -> str:
    system_instruction = None
    if messages and messages[0].get("role") == "system":
        system_instruction = messages[0].get("content")

    contents = build_gemini_contents(messages)
    
    # 3. New SDK requires snake_case for config keys
    config = {"temperature": 1.0, "max_output_tokens": 400}
    if system_instruction:
        config["system_instruction"] = system_instruction

    for attempt in range(3):
        try:
            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
            text = extract_gemini_text(response)
            if text and text != "I'm having trouble responding right now.":
                return text
            continue
        except Exception as exc:
            if attempt == 2:
                raise
            await asyncio.sleep(1)

    return "Gemini is a bit overloaded right now. Try again in a moment."


async def handle_ai_conversation(message: discord.Message, user_text: str) -> Optional[str]:
    conversation_key = get_conversation_key(message)
    user_text = clean_incoming_text(user_text)
    if not user_text:
        return None

    shared_history = conversation_memories.get(get_context_memory_key(message), [])
    personal_history = conversation_memories.get(conversation_key, [])
    history = shared_history[-10:] + personal_history[-10:]

    prompt = [
        {"role": "system", "content": BRAINROT_PROMPT},
        *history,
        {"role": "user", "content": user_text},
    ]

    try:
        async with message.channel.typing():
            ai_response = await create_gemini_response(prompt)

        add_to_memory(conversation_key, "user", user_text)
        add_to_memory(conversation_key, "assistant", ai_response)
        add_to_memory(get_context_memory_key(message), "user", user_text)
        add_to_memory(get_context_memory_key(message), "assistant", ai_response)
        return ai_response
    except Exception as exc:
        logger.exception("Gemini request failed")
        try:
            await message.channel.send(f"Gemini error: {type(exc).__name__}: {exc}")
        except Exception:
            # best-effort to notify user without raising further
            pass
        return None


@bot.event
async def on_ready() -> None:
    logger.info("Logged in as %s", bot.user)


@bot.event
async def on_message(message: discord.Message) -> None:
    try:
        if message.author.bot:
            return

        remember_live_context(message)
        referenced_message = await fetch_referenced_message(message)
        if referenced_message and referenced_message.author.id == bot.user.id:
            user_text = clean_incoming_text(message.content)
            if user_text:
                response = await handle_ai_conversation(message, user_text)
                if response:
                    await message.channel.send(response)
                return

        user_text = clean_incoming_text(message.content)
        target_channel = resolve_target_channel(message)
        is_direct_mention = bot.user in message.mentions
        is_channel_question = bool(
            re.search(
                r"\b(what|who|where|when|why|how|summarize|tell|read|say|happen|happened|discuss|discussed)\b",
                user_text.lower(),
            )
        )

        if (is_direct_mention or is_channel_question) and target_channel and target_channel.id != message.channel.id:
            perms = None
            if message.guild:
                perms = target_channel.permissions_for(message.guild.me)

            if not perms or not perms.view_channel or not perms.read_message_history:
                await message.channel.send(
                    "I don't have permission to read that channel's history."
                )
                return

            recent = []
            try:
                async for m in target_channel.history(limit=50, oldest_first=True):
                    if m.author.bot:
                        continue
                    text = m.content or ""
                    if not text.strip():
                        continue
                    recent.append(f"{m.author.display_name}: {text}")
            except discord.Forbidden:
                await message.channel.send(
                    "I don't have permission to read that channel's history."
                )
                return
            except Exception as exc:
                logger.exception("Failed to fetch referenced channel history")
                await message.channel.send(f"Failed to fetch messages: {exc}")
                return

            if recent:
                context_block = "\n".join(recent[-50:])
                user_text = (
                    f"Context from #{target_channel.name}:\n"
                    f"{context_block}\n\nQuestion: {user_text}"
                )
                logger.info(
                    "Included channel context from %s for message from %s",
                    target_channel.name,
                    message.author,
                )
            else:
                user_text = (
                    f"I couldn't find recent user messages in #{target_channel.name}.\n\nQuestion: {user_text}"
                )
                logger.info(
                    "No recent messages found in %s for %s",
                    target_channel.name,
                    message.author,
                )

            if user_text:
                response = await handle_ai_conversation(message, user_text)
                if response:
                    await message.channel.send(response)
                return
    except Exception:
        logger.exception("Unhandled exception in on_message")
        try:
            await message.channel.send("Sorry, I encountered an error while processing your message.")
        except Exception:
            pass
        return

    await bot.process_commands(message)


@bot.command()
async def tung(ctx: commands.Context, *, text: Optional[str] = None) -> None:
    if not text:
        referenced_message = await fetch_referenced_message(ctx.message)
        if referenced_message:
            text = referenced_message.content

    if not text:
        await ctx.send(
            "Reply to a message with `!tung`, or type `!tung <your text>`."
        )
        return

    response = await handle_ai_conversation(ctx.message, text)
    if response:
        await ctx.send(response)


@bot.command()
async def recap(
    ctx: commands.Context,
    channel: Optional[TextChannel] = None,
    limit: int = 25,
) -> None:
    if channel is None:
        channel = ctx.channel

    if ctx.guild is not None:
        if not (
            ctx.author.guild_permissions.manage_messages
            or ctx.author.guild_permissions.administrator
            or ctx.author == ctx.guild.owner
        ):
            await ctx.send("You don't have permission to request channel recaps.")
            return

    messages: List[str] = []
    try:
        async for m in channel.history(limit=limit, oldest_first=True):
            if m.author.bot:
                continue
            content = m.content or ""
            if not content:
                continue
            messages.append(f"{m.author.display_name}: {content}")
    except discord.Forbidden:
        await ctx.send("I don't have permission to read that channel's history.")
        return
    except Exception as exc:
        logger.exception("Failed to fetch channel history")
        await ctx.send(f"Failed to fetch messages: {exc}")
        return

    if not messages:
        await ctx.send("No user messages found to summarize.")
        return

    combined = "\n".join(messages[-limit:])
    system_prompt = (
        "You are an assistant that summarizes Discord channel conversations. Provide a concise summary and highlight any notable messages or action items."
    )
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarize the following messages from #{channel.name}:\n\n{combined}"},
    ]

    try:
        async with channel.typing():
            summary = await create_gemini_response(prompt)
        await ctx.send(f"Summary for {channel.mention}:\n{summary}")
    except Exception as exc:
        logger.exception("Gemini request failed for recap")
        await ctx.send(f"AI summarize failed: {exc}")


@bot.command()
async def remember(
    ctx: commands.Context,
    channel: Optional[TextChannel] = None,
    limit: int = 100,
) -> None:
    if channel is None:
        channel = ctx.channel

    if ctx.guild is None or channel.guild is None or channel.guild != ctx.guild:
        await ctx.send("I can only load history from the current server.")
        return

    try:
        added = await remember_past_channel_history(channel, limit=limit)
    except discord.Forbidden:
        await ctx.send("I don't have permission to read that channel's history.")
        return
    except Exception as exc:
        logger.exception("Failed to ingest history from channel %s", channel)
        await ctx.send(f"Failed to ingest channel history: {exc}")
        return

    if added == 0:
        await ctx.send(f"No past messages were added from {channel.mention}.")
    else:
        await ctx.send(f"Saved {added} past messages from {channel.mention} into memory.")


if __name__ == "__main__":
    bot.run(discord_token)