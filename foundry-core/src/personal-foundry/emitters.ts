import type {
  ConnectorEmitInput,
  ConnectorEmitOutput,
  OntologyEventRecord,
  OntologyEntityRecord,
  OntologyRelationRecord,
  PersonalEventBus,
} from "./interfaces.js";

function toEventPayload(
  payload: OntologyEntityRecord | OntologyRelationRecord | OntologyEventRecord,
  kind: "entity" | "relation" | "event"
): Record<string, unknown> {
  return {
    kind,
    ...payload,
  };
}

export async function emitOntologyBatchToEventBus(
  input: ConnectorEmitInput,
  bus: PersonalEventBus,
  now: () => string
): Promise<ConnectorEmitOutput> {
  const emittedAt = now();
  const eventIds: string[] = [];
  let index = 0;

  for (const entity of input.mapped.entities) {
    const out = await bus.publish({
      type: `entity:${entity.type}`,
      occurredAt: emittedAt,
      sourceSystem: entity.sourceSystem,
      actorId: input.actorId,
      classification: entity.classification,
      payload: toEventPayload(entity, "entity"),
      id: `${input.connectorRunId}.entity.${index++}`,
    });
    eventIds.push(out.id);
  }

  for (const relation of input.mapped.relations) {
    const out = await bus.publish({
      type: `relation:${relation.type}`,
      occurredAt: emittedAt,
      sourceSystem: relation.sourceSystem,
      actorId: input.actorId,
      classification: relation.classification,
      payload: toEventPayload(relation, "relation"),
      id: `${input.connectorRunId}.relation.${index++}`,
    });
    eventIds.push(out.id);
  }

  for (const event of input.mapped.events) {
    const out = await bus.publish({
      type: `event:${event.type}`,
      occurredAt: event.occurredAt,
      sourceSystem: event.sourceSystem,
      actorId: event.actorId ?? input.actorId,
      classification: event.classification,
      payload: toEventPayload(event, "event"),
      id: `${input.connectorRunId}.event.${index++}`,
    });
    eventIds.push(out.id);
  }

  return {
    eventIds,
    emittedAt,
  };
}
