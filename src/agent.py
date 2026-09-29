import asyncio
import logging
import os
import textwrap

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    ToolError,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics

from consultations import (
    ConsultationRequest,
    ConsultationRequestStore,
    InMemoryConsultationRequestStore,
    InvalidConsultationRequestError,
    SqliteConsultationRequestStore,
)

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(
        self,
        *,
        request_store: ConsultationRequestStore | None = None,
        session_id: str = "local-session",
    ) -> None:
        self._request_store = request_store or InMemoryConsultationRequestStore()
        self._session_id = session_id
        super().__init__(
            # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
            # See all available models at https://docs.livekit.io/agents/models/llm/
            llm=inference.LLM(model="google/gemma-4-31b-it"),
            # To use a realtime model instead of a voice pipeline, replace the LLM
            # with a RealtimeModel and remove the STT/TTS from the AgentSession
            # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/)
            # 1. Install livekit-agents[openai]
            # 2. Set OPENAI_API_KEY in .env.local
            # 3. Add `from livekit.plugins import openai` to the top of this file
            # 4. Replace the llm argument with:
            #     llm=openai.realtime.RealtimeModel(voice="marin")
            instructions=textwrap.dedent(
                """\
                You are the voice receptionist for BrightPath Consulting.

                Help callers learn about our services and arrange consultations.

                Be friendly, professional, and concise.
                Ask only one question at a time.
                Never invent prices, availability, or company policies.
                If you do not know something, explain that a team member will follow up.

                Collect a caller's name, phone number, and reason for contacting us
                when they want a consultation or callback.

                Because this is a voice conversation, use plain spoken language.
                Do not use markdown, lists, emojis, or long explanations.

                When a caller asks about consultation availability, always use the
                check_consultation_availability tool. Treat the tool response as the
                source of truth. Do not invent or promise availability.
                """
            ),
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."
    @function_tool
    async def get_business_information(
        self,
        context: RunContext,
        topic: str,
    ) -> str:
        """Get official information about BrightPath Consulting."""

        topic = topic.lower()

        if "service" in topic:
            return (
                "BrightPath Consulting provides website development "
                "and technology consulting."
            )

        if "hour" in topic or "open" in topic:
            return "BrightPath Consulting is open Monday through Friday, from nine AM to six PM."

        if "location" in topic or "where" in topic:
            return "BrightPath Consulting provides services remotely."

        if "consultation" in topic or "appointment" in topic:
            return "Consultations are available by appointment."

        return (
            "I do not have that information yet. "
            "A BrightPath team member can follow up."
        )

    @function_tool
    async def capture_customer_request(
        self,
        context: RunContext,
        name: str,
        phone: str,
        request: str,
    ) -> str:
        """Save a callback request after collecting complete caller details.

        Call this only after the caller supplies a name, phone number, and a
        specific reason for contacting BrightPath. Repeating the same call is
        safe and does not create a duplicate request.

        Args:
            name: Caller's full name.
            phone: Callback number with 7 to 15 digits, optionally including a
                leading country code.
            request: Concise reason for the callback or consultation.
        """

        try:
            customer_request = ConsultationRequest.create(
                session_id=self._session_id,
                name=name,
                phone=phone,
                request=request,
            )
        except InvalidConsultationRequestError as error:
            raise ToolError(str(error)) from error

        # A caller interruption must not cancel a write after it has begun. A
        # retry is safe because the persistence boundary enforces request_id.
        if context is not None:
            context.disallow_interruptions()

        try:
            created = await asyncio.wait_for(
                self._request_store.create(customer_request),
                timeout=5,
            )
        except TimeoutError as error:
            raise ToolError(
                "The callback system is taking too long. Tell the caller the "
                "request was not confirmed and offer to try again."
            ) from error
        except Exception as error:
            logger.exception(
                "failed to persist consultation request",
                extra={"request_id": customer_request.request_id},
            )
            raise ToolError(
                "The callback system is unavailable. Tell the caller the "
                "request was not confirmed and offer a human follow-up path."
            ) from error

        if not created:
            return (
                f"Thank you, {customer_request.name}. This request was already "
                "recorded, so I did not create a duplicate."
            )

        return (
            f"Thank you, {customer_request.name}. I recorded your request. "
            "A BrightPath team member can follow up with you."
        )

    @function_tool
    async def check_consultation_availability(
        self,
        context: RunContext,
        requested_day: str,
    ) -> str:
        """Check available BrightPath consultation times for a requested weekday."""

        day = requested_day.strip().lower()

        availability = {
            "monday": "Monday has consultation times at 10 AM and 2 PM.",
            "tuesday": "Tuesday has consultation times at 11 AM and 4 PM.",
            "wednesday": "Wednesday has a consultation time at 10 AM.",
            "thursday": "Thursday has consultation times at 2 PM and 5 PM.",
            "friday": "Friday has consultation times at 11 AM and 3 PM.",
        }

        for weekday, response in availability.items():
            if weekday in day:
                return response

        return (
            "I need a specific weekday to check availability. "
            "Please tell me which day you prefer."
        )


def _request_store_from_environment() -> ConsultationRequestStore:
    database_path = os.getenv("BRIGHTPATH_DB_PATH")
    if database_path:
        return SqliteConsultationRequestStore(database_path)
    return InMemoryConsultationRequestStore()


request_store = _request_store_from_environment()
server = AgentServer()


@server.rtc_session(agent_name="brightpath-receptionist")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using AssemblyAI, Fish Audio, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-3-5-pro", language="en"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="fishaudio/s2.1-pro", voice="fa4c9eb3dccc4806b382b40d61c6b10a"
        ),
        turn_handling=TurnHandlingOptions(
            # The LiveKit turn detector determines when the user is done speaking and the agent should respond.
            # TurnDetector is an end-of-turn model that listens to the user's audio directly, combining
            # semantic understanding with acoustic cues (intonation, pitch, rhythm) for state-of-the-art accuracy.
            # AgentSession supplies the required VAD automatically.
            # See more at https://docs.livekit.io/agents/build/turns
            turn_detection=inference.TurnDetector(),
            # Adaptive interruptions use the turn detector to tell a real interruption from a
            # backchannel like "mhm" or "right", so the agent keeps talking through the latter.
            interruption={"mode": "adaptive"},
            # allow the LLM to generate a response while waiting for the end of turn
            # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
            preemptive_generation={"enabled": True},
        ),
        # Expressive mode injects the TTS provider's markup guide into the LLM prompt, so the model
        # emits inline delivery tags (emotion, pacing, non-verbal sounds) that the TTS renders and
        # the transcript never shows. Requires a TTS model that supports markup, such as the Fish
        # Audio model above.
        expressive=True,
    )

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(request_store=request_store, session_id=ctx.room.name),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = anam.AvatarSession(
    #     persona_config=anam.PersonaConfig(
    #         name="...",
    #         avatarId="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/anam
    #     ),
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
